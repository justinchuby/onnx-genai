//! Checked-in graph fixtures must remain reviewable ONNX TextFormat.

use std::path::Path;
use std::process::Command;

const MIN_TEXTPROTO_FIXTURES: usize = 200;
const SENTINEL_FIXTURES: &[&str] = &[
    "crates/onnx-genai-engine/tests/fixtures/model-package-cpu/cpu/model.onnx.textproto",
    "crates/onnx-genai-metadata/tests/fixtures/validator_package/identity.onnx.textproto",
    "crates/onnx-runtime-ep-cuda-plugin/tests/fixtures/unique_all_outputs/model.onnx.textproto",
    "crates/onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto",
    "tests/fixtures/onnx_genai_workflows/diffusion/denoiser/model.onnx.textproto",
    "tests/fixtures/tiny-llm/model.onnx.textproto",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtensionClass {
    Textproto,
    BinaryOnnx,
    Other,
    Invalid,
}

fn normalized_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// Mirror the authoritative extension-normalization contract documented by
/// `.github/scripts/ci_change_scope.sh`:
///
/// 1. retain the slash-normalized path for equality and diagnostics;
/// 2. strip only trailing ASCII space, tab, CR, LF, vertical tab, and form feed
///    from a separate classification copy;
/// 3. reject an empty path or terminal filename after stripping;
/// 4. ASCII-lowercase the copy and inspect only its terminal extension.
///
/// Interior whitespace and trailing dots are deliberately not normalized.
fn extension_class(path: &str) -> ExtensionClass {
    let normalized = normalized_path(path);
    let extension_path = normalized
        .trim_end_matches([' ', '\t', '\r', '\n', '\u{000b}', '\u{000c}'])
        .to_ascii_lowercase();
    if extension_path.is_empty() || extension_path.ends_with('/') {
        ExtensionClass::Invalid
    } else if extension_path.ends_with(".textproto") {
        ExtensionClass::Textproto
    } else if extension_path.ends_with(".onnx") {
        ExtensionClass::BinaryOnnx
    } else {
        ExtensionClass::Other
    }
}

fn is_textproto(path: &str) -> bool {
    extension_class(path) == ExtensionClass::Textproto
}

fn is_binary_onnx(path: &str) -> bool {
    extension_class(path) == ExtensionClass::BinaryOnnx
}

fn fixture_inventory_errors(paths: &[String]) -> Vec<String> {
    // Extension policy is case-insensitive. Mixed-case textproto remains
    // reviewable text and is allowed, but it counts toward (and therefore must
    // pass through) this census. Every casing of a terminal `.onnx` is binary
    // and forbidden.
    let normalized_paths = paths
        .iter()
        .map(|path| normalized_path(path))
        .collect::<Vec<_>>();
    let textproto_count = normalized_paths
        .iter()
        .filter(|path| is_textproto(path))
        .count();
    let binaries = normalized_paths
        .iter()
        .filter(|path| is_binary_onnx(path))
        .cloned()
        .collect::<Vec<_>>();
    let invalid = normalized_paths
        .iter()
        .filter(|path| extension_class(path) == ExtensionClass::Invalid)
        .cloned()
        .collect::<Vec<_>>();
    let missing_sentinels = SENTINEL_FIXTURES
        .iter()
        .filter(|sentinel| !normalized_paths.iter().any(|path| path == **sentinel))
        .copied()
        .collect::<Vec<_>>();

    let mut errors = Vec::new();
    if textproto_count < MIN_TEXTPROTO_FIXTURES {
        errors.push(format!(
            "tracked textproto fixture inventory collapsed to {textproto_count}; expected at least \
             {MIN_TEXTPROTO_FIXTURES}. Check the repository root and git pathspecs"
        ));
    }
    if !missing_sentinels.is_empty() {
        errors.push(format!(
            "tracked textproto fixture census is missing sentinel(s): {}. A fixture path or the \
             census root/pathspec drifted",
            missing_sentinels.join(", ")
        ));
    }
    if !binaries.is_empty() {
        errors.push(format!(
            "checked-in ONNX graphs must use *.onnx.textproto, found binary fixture(s): {}",
            binaries.join(", ")
        ));
    }
    if !invalid.is_empty() {
        errors.push(format!(
            "tracked path(s) have an empty/all-ASCII-whitespace terminal filename after extension \
             normalization and cannot be classified safely: {}",
            invalid.join(", ")
        ));
    }
    errors
}

fn tracked_files(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        // Enumerate first, classify second. Git pathspec case behavior differs
        // across filesystems; a lowercase pathspec can miss `model.ONNX` on a
        // case-sensitive runner and make the binary prohibition vacuous.
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .expect("git must be available to audit checked-in fixtures");
    assert!(
        output.status.success(),
        "git fixture audit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .split('\0')
        .filter(|path| !path.is_empty())
        .filter(|path| root.join(path).is_file())
        .map(normalized_path)
        .collect()
}

#[test]
fn checked_in_graph_fixtures_are_textproto_and_census_is_intact() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = tracked_files(&root);
    let errors = fixture_inventory_errors(&paths);
    assert!(
        errors.is_empty(),
        "checked-in ONNX fixture census failed:\n  - {}",
        errors.join("\n  - ")
    );
}

fn healthy_fixture_inventory() -> Vec<String> {
    let mut paths = SENTINEL_FIXTURES
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    paths.extend(
        (paths.len()..MIN_TEXTPROTO_FIXTURES)
            .map(|index| format!("tests/fixtures/synthetic-{index}/model.onnx.textproto")),
    );
    paths
}

#[test]
fn fixture_census_rejects_empty_enumeration() {
    assert!(
        !fixture_inventory_errors(&[]).is_empty(),
        "an empty fixture enumeration must fail closed"
    );
}

#[test]
fn fast_fixture_census_rejects_docs_binary_onnx() {
    let mut fixtures = healthy_fixture_inventory();
    for path in [
        "docs/foo/model.onnx",
        "docs/foo/model.ONNX",
        "docs/foo/model.OnNx",
        r"docs\foo\model.ONNX",
        "docs/foo/model.ONNX ",
        "docs/foo/model.OnNx\t",
        "docs/foo/model.onnx\r",
        "docs/foo/model.onnx\n",
        "docs/foo/model.onnx\u{000b}",
        "docs/foo/model.onnx\u{000c}",
        "docs/foo/model .ONNX",
    ] {
        fixtures.push(path.to_owned());
        let diagnostic_path = normalized_path(path);
        assert!(
            fixture_inventory_errors(&fixtures)
                .iter()
                .any(|error| error.contains(&diagnostic_path)),
            "tracked binary ONNX must be rejected after extension normalization: {path:?}"
        );
        fixtures.pop();
    }
    fixtures.push("docs/foo/model.ONNX.md".to_owned());
    assert!(
        fixture_inventory_errors(&fixtures).is_empty(),
        "only the terminal extension decides classification"
    );
}

#[test]
fn change_scope_keeps_docs_textproto_in_fast_census() {
    let mut fixtures = healthy_fixture_inventory();
    for path in [
        "docs/foo/model.onnx.textproto",
        "docs/foo/model.ONNX.TEXTPROTO",
        "docs/foo/model.OnNx.TeXtPrOtO",
        r"docs\foo\model.ONNX.TEXTPROTO",
        "docs/foo/model.ONNX.TEXTPROTO ",
        "docs/foo/model.OnNx.TeXtPrOtO\t",
        "docs/foo/model.onnx.textproto\r",
        "docs/foo/model.onnx.textproto\n",
        "docs/foo/model.onnx.textproto\u{000b}",
        "docs/foo/model.onnx.textproto\u{000c}",
    ] {
        fixtures.push(path.to_owned());
        assert!(
            fixture_inventory_errors(&fixtures).is_empty(),
            "textproto casing is allowed, but change-scope must still run this census: {path}"
        );
        fixtures.pop();
    }

    #[cfg(unix)]
    {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let classifier = root.join(".github/scripts/ci_change_scope.sh");
        for (path, expected_docs_only) in [
            ("docs/foo/model.ONNX ", false),
            ("docs/foo/model.OnNx\t", false),
            ("docs/foo/model.ONNX.TEXTPROTO ", false),
            ("docs/foo/model.OnNx.TeXtPrOtO\t", false),
            ("docs/foo/model.ONNX.md", true),
            ("docs/foo/model .ONNX", false),
            ("", false),
            ("   ", false),
            ("docs/foo/\t", false),
        ] {
            let output = Command::new("bash")
                .args([
                    "-c",
                    ". \"$1\"; if ci_is_docs_path \"$2\"; then printf docs; else printf code; fi",
                    "_",
                ])
                .arg(&classifier)
                .arg(path)
                .output()
                .expect("bash must run the CI classifier parity check");
            assert!(
                output.status.success(),
                "bash classifier failed for {path:?}: status={:?} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
            let bash_docs_only = output.stdout == b"docs";
            assert_eq!(
                bash_docs_only, expected_docs_only,
                "Bash classifier disagreed with the documented policy for {path:?}"
            );
            let rust_significant = matches!(
                extension_class(path),
                ExtensionClass::Textproto | ExtensionClass::BinaryOnnx | ExtensionClass::Invalid
            );
            assert_eq!(
                rust_significant, !expected_docs_only,
                "Rust census disagreed with Bash for {path:?}"
            );
        }
    }

    let mut fixtures = healthy_fixture_inventory();
    for path in ["", "   ", "\t\r\n\u{000b}\u{000c}", "docs/foo/\t"] {
        fixtures.push(path.to_owned());
        assert!(
            fixture_inventory_errors(&fixtures)
                .iter()
                .any(|error| error.contains("cannot be classified safely")),
            "empty/all-whitespace terminal filename must fail closed: {path:?}"
        );
        fixtures.pop();
    }
    assert_eq!(
        extension_class("docs/foo/model .ONNX"),
        ExtensionClass::BinaryOnnx,
        "interior whitespace must be preserved"
    );
    assert_eq!(
        extension_class("docs/foo/model.ONNX."),
        ExtensionClass::Other,
        "trailing dots are outside the ASCII-whitespace policy"
    );
}

#[test]
fn fixture_census_rejects_sentinel_path_drift() {
    let mut fixtures = healthy_fixture_inventory();
    fixtures.retain(|path| path != SENTINEL_FIXTURES[0]);
    fixtures.push("tests/fixtures/renamed/model.onnx.textproto".to_owned());
    assert_eq!(
        fixtures.iter().filter(|path| is_textproto(path)).count(),
        MIN_TEXTPROTO_FIXTURES,
        "the mutation must preserve the count so only the sentinel detects it"
    );
    assert!(
        fixture_inventory_errors(&fixtures)
            .iter()
            .any(|error| error.contains(SENTINEL_FIXTURES[0])),
        "renaming or removing a sentinel path must fail even above the count floor"
    );
}

/// Substring expansions that pass a negative *length* -- `${var::-1}` and the
/// non-empty-offset spelling `${var:1:-1}` alike -- reported as `(line number,
/// trimmed line)`.
///
/// Deliberately not reported, because every bash this repository runs on accepts
/// them: negative *offsets* (`${var: -1}`), suffix removal (`${var%?}`), length
/// (`${#var}`), and the `${var:-word}` / `:=` / `:?` / `:+` word operators. Those
/// are separated from a substring expansion exactly as bash separates them -- an
/// operator character immediately after the first colon, with no space.
///
/// Two known imprecisions, both chosen so the check over-reports rather than
/// under-reports: an expansion nested inside another (`${x:${y}:-1}`) or split
/// across a line continuation is not parsed, and an inert `${var::-1}` inside a
/// single-quoted string would still be flagged. Over-reporting is a rewordable
/// build failure; under-reporting re-reddens a lane nobody is required to look at.
fn negative_length_substring_expansions(source: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    for (index, line) in source.lines().enumerate() {
        // A whole-line comment cannot be expanded, so it is not a hit. Anything
        // after code on the same line still is: deciding where a trailing `#`
        // starts a comment needs a shell lexer.
        if line.trim_start().starts_with('#') {
            continue;
        }
        if line
            .match_indices("${")
            .any(|(open, _)| expansion_has_negative_length(&line[open + 2..]))
        {
            hits.push((index + 1, line.trim().to_owned()));
        }
    }
    hits
}

/// `body` starts just past a `${`. Reads up to the first `}` and decides whether
/// that expansion supplies a negative substring length.
fn expansion_has_negative_length(body: &str) -> bool {
    let body = match body.find('}') {
        Some(end) => &body[..end],
        None => return false,
    };
    let Some(first) = body.find(':') else {
        return false;
    };
    // `${var:-word}`, `:=`, `:?`, `:+` are word operators, not substring bounds.
    // Bash requires the operator to abut the colon, so `${var: -1}` is a substring
    // expansion with a negative offset and is left alone.
    if matches!(
        body[first + 1..].chars().next(),
        Some('-' | '=' | '?' | '+')
    ) {
        return false;
    }
    let Some(second) = body[first + 1..].find(':') else {
        return false;
    };
    body[first + 1 + second + 1..].trim_start().starts_with('-')
}

/// macOS ships bash 3.2 -- the `macos-*-arm64` runner images publish
/// `Bash 3.2.57(1)-release` -- and a negative length in substring expansion needs
/// bash >= 4.2. On 3.2, `${var::-1}` aborts with `substring expression < 0` and
/// exits 1, which surfaces as a classifier that failed rather than a classifier
/// that disagreed. The parity loop in
/// `change_scope_keeps_docs_textproto_in_fast_census` cannot see this: it runs
/// whatever `bash` is on `PATH`, and every Linux lane has bash >= 4.2. The only
/// lane that observed it is `Rust coverage (macOS arm64)`, which is not a required
/// check, so it sat red on `main`. This keeps the one construct that has actually
/// broken that lane visible from Linux. It checks that construct only; it is not a
/// bash 3.2 conformance proof for the script.
#[test]
fn ci_change_scope_avoids_bash_4_only_substring_expansion() {
    // Non-vacuity. Every fatal spelling must be flagged: the one that reddened the
    // lane, the non-empty-offset spelling that carries no `::-` at all, a spaced
    // variant, and a computed length. `${var:1:-1}` is the arm that matters most --
    // a matcher keyed on the literal `::-` passes this test while missing it, and
    // the required Linux lane never executes the script under bash 3.2 to notice.
    // Each was checked against both shells before being listed here: exit 1 with
    // `substring expression < 0` on 3.2, exit 0 on 5.2. `${arr[@]:1:-1}` is not in
    // the list despite being flagged -- a negative length on an array subscript is
    // an error on every bash, so it discriminates nothing.
    for fatal in [
        "x=\"${extension_path::-1}\"",
        "x=\"${extension_path:1:-1}\"",
        "x=\"${extension_path: 0: -1}\"",
        "x=\"${extension_path::-$n}\"",
        "x=\"${p::-1}\" # code carrying a trailing comment is still code",
    ] {
        assert_eq!(
            negative_length_substring_expansions(fatal)
                .into_iter()
                .map(|(line, _)| line)
                .collect::<Vec<_>>(),
            vec![1],
            "must flag a negative substring length: {fatal}"
        );
    }

    // ...and every neighbour bash 3.2 accepts must be left alone. A build-blocking
    // check in a required lane earns its false positives one at a time.
    for benign in [
        "case \"${extension_path: -1}\" in",
        "x=\"${extension_path%?}\"",
        "local embedded=\"${2-}\"",
        "y=\"${target:-fallback}\"",
        "y=\"${target:-a:-1}\"",
        "echo \"${first}: ${second}: -1\"",
        "y=\"${target:=default}\"",
        "y=\"${target:?must be set}\"",
        "y=\"${target:+present}\"",
        "n=\"${#extension_path}\"",
        "x=\"${extension_path:1:2}\"",
        "x=\"${path//\\\\//}\"",
        "  # prose naming ${var::-1} is inert",
    ] {
        assert!(
            negative_length_substring_expansions(benign).is_empty(),
            "bash 3.2 accepts this, so it must not be flagged: {benign}"
        );
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let classifier = root.join(".github/scripts/ci_change_scope.sh");
    let source = std::fs::read_to_string(&classifier)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", classifier.display()));
    let hits = negative_length_substring_expansions(&source);
    assert!(
        hits.is_empty(),
        "{} uses a bash >= 4.2 negative substring length, which exits 1 under macOS bash 3.2: {hits:?}",
        classifier.display()
    );
}
