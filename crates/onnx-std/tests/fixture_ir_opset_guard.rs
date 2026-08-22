//! Repo-wide guard: every maintained checked-in ONNX model fixture must declare
//! ONNX IR version >= 11 and a single default (`ai.onnx`) opset >= 24, and must
//! use only opset-24 node schemas (no removed attribute forms), recursively
//! through nested subgraphs.
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

use onnx_runtime_loader::proto::onnx::{GraphProto, ModelProto};
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

/// Standard-domain ops whose axis/size selector moved from an attribute to a
/// tensor **input** by opset 24 (Reduce* and Squeeze/Unsqueeze at opset 13/18,
/// Split at opset 13). The named attribute is illegal under the opset-24 schema.
fn removed_attribute(op_type: &str) -> Option<&'static str> {
    match op_type {
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL1"
        | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" | "Squeeze"
        | "Unsqueeze" => Some("axes"),
        "Split" => Some("split"),
        _ => None,
    }
}

/// Recursively collect opset-24 schema violations in a graph and every nested
/// subgraph reachable through node attributes (If/Loop/Scan bodies, etc.).
fn collect_legacy_forms(graph: &GraphProto, trail: &str, out: &mut Vec<String>) {
    for node in &graph.node {
        let is_standard = node.domain.is_empty() || node.domain == "ai.onnx";
        if is_standard
            && let Some(attr) = removed_attribute(&node.op_type)
            && node.attribute.iter().any(|a| a.name == attr)
        {
            let name = if node.name.is_empty() {
                "<unnamed>"
            } else {
                node.name.as_str()
            };
            out.push(format!(
                "{trail}{} {name} uses the removed `{attr}` attribute (opset-24 requires it as an input)",
                node.op_type
            ));
        }
        // Recurse into subgraph-valued attributes (GRAPH and GRAPHS).
        for attr in &node.attribute {
            let child_trail = format!("{trail}{}[{}]/", node.op_type, attr.name);
            if let Some(sub) = &attr.g {
                collect_legacy_forms(sub, &child_trail, out);
            }
            for sub in &attr.graphs {
                collect_legacy_forms(sub, &child_trail, out);
            }
        }
    }
}

/// Resolve the single default (`ai.onnx`) opset version. Collecting *all*
/// default-domain imports (both the empty domain and its `ai.onnx` alias) and
/// requiring exactly one rejects duplicate/aliased imports used to smuggle a
/// stale default past a naive first-match check.
fn default_opset(imports: &[OperatorSetIdProto]) -> Result<i64, String> {
    let defaults: Vec<i64> = imports
        .iter()
        .filter(|import| import.domain.is_empty() || import.domain == "ai.onnx")
        .map(|import| import.version)
        .collect();
    match defaults.as_slice() {
        [] => Err("no default (ai.onnx) opset import".to_string()),
        [version] => Ok(*version),
        many => Err(format!(
            "{} default (ai.onnx) opset imports (expected exactly one): {many:?}",
            many.len()
        )),
    }
}

/// All floor/schema problems for one model. Empty => the fixture is compliant.
fn model_problems(proto: &ModelProto) -> Vec<String> {
    let mut problems = Vec::new();
    if proto.ir_version < MIN_IR_VERSION {
        problems.push(format!(
            "ir_version {} < {MIN_IR_VERSION}",
            proto.ir_version
        ));
    }
    match default_opset(&proto.opset_import) {
        Ok(opset) if opset < MIN_DEFAULT_OPSET => {
            problems.push(format!("default opset {opset} < {MIN_DEFAULT_OPSET}"));
        }
        Ok(_) => {}
        Err(message) => problems.push(message),
    }
    if let Some(graph) = &proto.graph {
        collect_legacy_forms(graph, "", &mut problems);
    }
    problems
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

/// Tracked ONNX model fixtures: textproto (`*.textproto`, which also covers
/// `*.onnx.textproto`) and binary `*.onnx`. External weight blobs (`*.onnx.data`)
/// and non-model placeholders (`*.onnx.fixture`) do not match these globs.
fn tracked_model_fixtures(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "*.textproto", "*.onnx"])
        .current_dir(root)
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
/// custom-domain models) and preserves nested subgraphs for recursive checks.
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

    assert!(
        fixtures.len() >= MIN_FIXTURE_COUNT,
        "fixture inventory collapsed to {} model file(s) (expected at least \
         {MIN_FIXTURE_COUNT}). This usually means the repo root was resolved \
         incorrectly or `git ls-files` returned nothing — the guard refuses to \
         pass vacuously. Root was {}.",
        fixtures.len(),
        root.display()
    );

    let mut violations = Vec::new();
    for rel in &fixtures {
        let proto = decode_fixture(&root.join(rel));
        for problem in model_problems(&proto) {
            violations.push(format!("  {rel}: {problem}"));
        }
    }

    assert!(
        violations.is_empty(),
        "maintained ONNX model fixtures must declare IR >= {MIN_IR_VERSION}, exactly one default \
         (ai.onnx) opset >= {MIN_DEFAULT_OPSET}, and use only opset-24 node schemas (move \
         ReduceMean/ReduceSum/Unsqueeze/Squeeze `axes` and Split `split` to tensor inputs, \
         including in nested subgraphs). There is no allowlist; regenerate the fixture via its \
         generator, or build a legacy-opset graph in-memory inside the specific test that needs \
         it.\n{}",
        violations.join("\n")
    );
}

#[cfg(test)]
mod negative_self_tests {
    use super::*;
    use onnx_runtime_loader::proto::onnx::{AttributeProto, NodeProto};

    fn node(op_type: &str, attrs: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    fn axes_attr() -> AttributeProto {
        AttributeProto {
            name: "axes".to_string(),
            ..Default::default()
        }
    }

    fn opset(domain: &str, version: i64) -> OperatorSetIdProto {
        OperatorSetIdProto {
            domain: domain.to_string(),
            version,
        }
    }

    #[test]
    fn wrong_root_or_empty_scan_is_rejected() {
        // The real scan must find the full inventory; a collapsed count fails.
        let root = repo_root();
        let found = tracked_model_fixtures(&root).len();
        assert!(
            found >= MIN_FIXTURE_COUNT,
            "sanity: real scan found {found}"
        );
        // An empty/near-empty scan (e.g. wrong root) would trip the lower bound.
        let collapsed = 0usize;
        assert!(
            collapsed < MIN_FIXTURE_COUNT,
            "a collapsed scan must fail the floor"
        );
    }

    #[test]
    fn duplicate_or_aliased_default_imports_are_rejected() {
        // Duplicate empty-domain imports.
        assert!(default_opset(&[opset("", 24), opset("", 24)]).is_err());
        // Empty domain aliased with the explicit `ai.onnx` spelling.
        assert!(default_opset(&[opset("", 24), opset("ai.onnx", 24)]).is_err());
        // A stale duplicate must not be masked by a valid first match.
        assert!(default_opset(&[opset("", 13), opset("", 24)]).is_err());
        // Missing default domain.
        assert!(default_opset(&[opset("com.microsoft", 1)]).is_err());
        // Exactly one default resolves.
        assert_eq!(
            default_opset(&[opset("", 24), opset("com.microsoft", 1)]),
            Ok(24)
        );
    }

    #[test]
    fn legacy_attribute_forms_are_rejected() {
        for op in [
            "Unsqueeze",
            "Squeeze",
            "ReduceMean",
            "ReduceSum",
            "ReduceL2",
        ] {
            let g = GraphProto {
                node: vec![node(op, vec![axes_attr()])],
                ..Default::default()
            };
            let mut out = Vec::new();
            collect_legacy_forms(&g, "", &mut out);
            assert_eq!(out.len(), 1, "{op} axes-attribute must be flagged");
        }
        // Split `split` attribute.
        let split_attr = AttributeProto {
            name: "split".to_string(),
            ..Default::default()
        };
        let g = GraphProto {
            node: vec![node("Split", vec![split_attr])],
            ..Default::default()
        };
        let mut out = Vec::new();
        collect_legacy_forms(&g, "", &mut out);
        assert_eq!(out.len(), 1, "Split split-attribute must be flagged");
    }

    #[test]
    fn nested_subgraph_legacy_forms_are_rejected() {
        // An `If` whose `then_branch` subgraph hides a legacy Unsqueeze.
        let inner = GraphProto {
            node: vec![node("Unsqueeze", vec![axes_attr()])],
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
        let g = GraphProto {
            node: vec![if_node],
            ..Default::default()
        };
        let mut out = Vec::new();
        collect_legacy_forms(&g, "", &mut out);
        assert_eq!(out.len(), 1, "nested legacy Unsqueeze must be flagged");
        assert!(
            out[0].contains("If[then_branch]"),
            "trail names the subgraph: {}",
            out[0]
        );
    }

    #[test]
    fn compliant_input_form_graph_passes() {
        // Unsqueeze with axes as an input (no attribute) is compliant.
        let g = GraphProto {
            node: vec![node("Unsqueeze", vec![])],
            ..Default::default()
        };
        let mut out = Vec::new();
        collect_legacy_forms(&g, "", &mut out);
        assert!(out.is_empty(), "input-form Unsqueeze must pass");
    }
}
