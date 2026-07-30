// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Static file serving for the demo dashboard at `GET /demo`.
//!
//! The files themselves are served by [`tower_http::services::ServeDir`], which
//! owns path-traversal safety, content-type detection and conditional requests.
//! This module supplies only the parts `ServeDir` does not:
//!
//! * locating the asset directory,
//! * an actionable response when it was not found,
//! * the `/demo` -> `/demo/` redirect.
//!
//! Serving the demo from the server itself is load-bearing rather than a
//! convenience: it makes the dashboard same-origin with the API, so a visitor
//! can open a plain running server and see the demo with no proxy, no build
//! step, and no cross-origin configuration.

use std::path::PathBuf;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
};

/// Directory searched for demo assets when no explicit path is configured,
/// resolved relative to the current working directory.
const DEFAULT_DEMO_ASSETS_DIR: &str = "examples/serving-dashboard";

/// Shown when `/demo` is requested but no asset directory was found.
///
/// Names the flag, the environment variable AND the default location: starting
/// the server from the wrong working directory is the most likely first-run
/// mistake, and a bare `404` gives the visitor nothing to act on.
const MISSING_ASSETS_MESSAGE: &str = concat!(
    "The demo dashboard assets were not found.\n\n",
    "The server looks for them in this order:\n",
    "  1. --demo-assets-dir <DIR>\n",
    "  2. ONNX_GENAI_DEMO_ASSETS_DIR=<DIR>\n",
    "  3. ./examples/serving-dashboard (relative to the working directory)\n\n",
    "A directory is used only if it contains a readable, non-empty index.html. \
     A directory that exists but has no entry page is REFUSED rather than \
     served, because /demo publishes what it is pointed at and this server \
     needs no credentials.\n\n",
    "Either start the server from the repository root, or pass the directory \
     explicitly:\n",
    "  onnx-genai-server --model <MODEL_DIR> --demo-assets-dir examples/serving-dashboard\n",
);

/// The file a directory must be able to serve before it is treated as the demo.
///
/// This is the page `/demo/` resolves to, so a directory without it cannot
/// serve the dashboard no matter what else it contains.
const DEMO_ENTRY_PAGE: &str = "index.html";

/// Whether this directory can actually serve the dashboard.
///
/// Deliberately asks what the directory CAN DO rather than whether it exists.
/// `is_dir()` alone is satisfied by `/`, `$HOME` or `/etc`, and the `/demo`
/// mount publishes whatever it is pointed at on an unauthenticated server — so
/// the check that keeps a typo from turning into disclosure has to be evidence
/// that this is the dashboard, not evidence that the path is real.
///
/// The entry page must be a non-empty regular file we can read: a directory
/// named `index.html`, a dangling symlink, an unreadable file and a zero-byte
/// placeholder all satisfy "exists" and none of them can serve the demo. Using
/// the metadata we already had to fetch avoids claiming more than we checked —
/// this does not parse the HTML, and does not pretend to.
fn directory_can_serve_the_dashboard(candidate: &std::path::Path) -> bool {
    if !candidate.is_dir() {
        return false;
    }
    // `metadata` follows symlinks, so a link to a real page is accepted and a
    // dangling one is not — which matches what `ServeDir` will do later.
    std::fs::metadata(candidate.join(DEMO_ENTRY_PAGE))
        .is_ok_and(|entry| entry.is_file() && entry.len() > 0)
}

/// Resolves the demo asset directory from an explicit override, the environment,
/// then a working-directory-relative default.
///
/// Returns `None` when no candidate can serve the dashboard, so a server
/// started outside the repo still boots — `/demo` then reports the problem
/// instead of the process refusing to start over an optional feature.
///
/// A candidate that exists but cannot serve the dashboard is rejected rather
/// than mounted, because mounting it publishes its contents. The three sources
/// are NOT tried in turn on failure: an explicit `--demo-assets-dir` that is
/// wrong is a mistake to report, not a reason to silently serve a different
/// directory than the operator named.
pub fn resolve_demo_assets_dir(explicit: Option<PathBuf>) -> Option<PathBuf> {
    let candidate = explicit
        .or_else(|| std::env::var_os("ONNX_GENAI_DEMO_ASSETS_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DEMO_ASSETS_DIR));
    directory_can_serve_the_dashboard(&candidate).then_some(candidate)
}

/// Fallback for every `/demo` route when no asset directory was configured.
pub(crate) async fn missing_assets() -> Response {
    (StatusCode::NOT_FOUND, MISSING_ASSETS_MESSAGE).into_response()
}

/// File extensions the demo dashboard actually loads in a browser.
///
/// An allowlist rather than a denylist because the asset directory is a
/// working source tree, not a build output: it gains files continuously and a
/// denylist would have to be updated by whoever adds the next kind, which is
/// the person least likely to be thinking about disclosure.
const SERVABLE_EXTENSIONS: [&str; 9] = [
    "html", "js", "mjs", "css", "json", "svg", "png", "ico", "woff2",
];

/// Whether `ServeDir` should be allowed to answer for this `/demo` path.
///
/// Returns true for paths outside `/demo` so this predicate can be used from a
/// router-wide layer without opinion on the rest of the API.
fn demo_path_is_servable(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("/demo/") else {
        return true;
    };
    // A directory request; `ServeDir` resolves it to `index.html`, which is on
    // the list.
    if rest.is_empty() || rest.ends_with('/') {
        return true;
    }
    let name = rest.rsplit('/').next().unwrap_or(rest);
    // Dotfiles are refused by segment, not by name. `ServeDir` has NO dotfile
    // rule of its own -- measured, not assumed: with the extension allowlist in
    // place `.secret.json` and `.vscode/settings.json` both answered 200 with
    // their contents, and `.env`, `.npmrc` and `.git/config` were refused only
    // incidentally, because their extensions are not on the list. That is a
    // refusal by coincidence, and it inverts the moment someone adds `json` to
    // a dotted config directory. Checking every segment rather than the final
    // name is what stops `.git/config` and `.vscode/settings.json`, where the
    // secret is the DIRECTORY.
    if rest.split('/').any(|segment| segment.starts_with('.')) {
        return false;
    }
    // Test files are `.js` and would pass the extension check. They are not
    // secret, but they are the largest source of confusing 200s for anyone
    // poking at the demo, and nothing in the page loads them.
    if name.ends_with(".test.js") || name.ends_with(".test.mjs") {
        return false;
    }
    let Some((_, extension)) = name.rsplit_once('.') else {
        return false;
    };
    SERVABLE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
}

/// Serves only what the page loads, out of a directory that is a source tree.
///
/// The asset directory is `examples/serving-dashboard`, which at the time of
/// writing held 101 files: 15 markdown documents including an unpublished
/// architecture and security review, two shell scripts, three Python scripts,
/// and 47 test files. Every one of them answered `200` on the demo origin. The
/// review document alone was 21KB of our own findings, served to any visitor
/// of a machine running the demo.
///
/// This is not a traversal bug -- `ServeDir` handles traversal correctly and
/// these files are genuinely inside the directory it was pointed at. The defect
/// is that "the directory the assets live in" and "the set of files the demo
/// should publish" were assumed to be the same set, and they are not.
pub(crate) async fn restrict_demo_assets(request: Request<Body>, next: Next) -> Response {
    if !demo_path_is_servable(request.uri().path()) {
        // A plain 404, identical to a path that does not exist: distinguishing
        // "refused" from "absent" would confirm the file is there.
        return StatusCode::NOT_FOUND.into_response();
    }
    next.run(request).await
}

/// The parts of the demo policy that do not depend on where the server is bound.
///
/// `script-src` needs no `'unsafe-inline'`: the page loads exactly one external
/// module and has no inline script, which was verified rather than assumed.
///
/// `style-src` keeps `'unsafe-inline'` as a stated concession. The markup
/// carries no inline styles today, but the dashboard is another agent's tree
/// under a change freeze, and a policy that blanks a panel on stage would be
/// far worse than the risk it removes. Narrow it when the freeze lifts.
const DEMO_CSP_PREFIX: &str = concat!(
    "default-src 'self'; ",
    "script-src 'self'; ",
    "style-src 'self' 'unsafe-inline'; ",
    "img-src 'self' data:; ",
    "object-src 'none'; ",
    "base-uri 'self'; ",
    "frame-ancestors 'none'; ",
    "connect-src 'self'",
);

/// Builds the demo policy for the host the page was actually requested from.
///
/// `connect-src` is the directive doing the security work, and it is a
/// browser-layer mitigation for the origin-injection defect: the dashboard
/// takes its server origins from a query parameter, so a crafted link can aim
/// our page at a third party and render fabricated numbers under our own
/// provenance badges. Confining fetches to the page's own host means the
/// parser AND the browser must both be defeated, not just the parser.
///
/// **Same host, any port** -- which is the topology, not laxity. The demo runs
/// two servers on one host and the page served by one legitimately polls the
/// other. CSP treats a differing port as a differing origin, so a bare
/// `'self'` here would break the demo rather than secure it.
///
/// Derived from the request rather than from a configured bind address for two
/// reasons: the bind address was deleted along with the path-disclosure
/// conditional it existed to feed, and a hardcoded loopback literal silently
/// breaks the dashboard the first time anyone overrides `BIND_HOST` -- failing
/// closed, on stage, with a console error as the only symptom.
///
/// The `Host` header is client-controlled, so it is validated to a bare
/// hostname before use; anything else falls back to `'self'` alone. This
/// widens the policy only to the host family the user already navigated to,
/// and it can never inject a second directive.
fn demo_csp_for_host(host: Option<&str>) -> String {
    let Some(host) = host.and_then(sanitised_host) else {
        return DEMO_CSP_PREFIX.to_string();
    };
    format!("{DEMO_CSP_PREFIX} http://{host}:* https://{host}:*")
}

/// Strips any port and rejects anything that is not a bare hostname or IP.
///
/// The rejection is what keeps a header value from carrying a `;` and adding a
/// directive of the attacker's choosing.
fn sanitised_host(host: &str) -> Option<&str> {
    // IPv6 literals are bracketed and would need their own escaping rules to
    // embed safely; loopback IPv6 is already covered by `'self'`.
    if host.starts_with('[') {
        return None;
    }
    let name = host.split(':').next()?;
    let valid = !name.is_empty()
        && name.len() <= 253
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-');
    valid.then_some(name)
}

/// Attaches the demo policy and `nosniff` to demo responses.
///
/// Scoped to `/demo` rather than applied router-wide because a policy on a JSON
/// API response is inert, and a header that does nothing invites the belief
/// that the whole surface is covered.
pub(crate) async fn demo_security_headers(request: Request<Body>, next: Next) -> Response {
    let is_demo = request.uri().path().starts_with("/demo");
    let policy = is_demo.then(|| {
        demo_csp_for_host(
            request
                .headers()
                .get(axum::http::header::HOST)
                .and_then(|host| host.to_str().ok()),
        )
    });
    let mut response = next.run(request).await;
    if let Some(policy) = policy {
        if let Ok(value) = axum::http::HeaderValue::from_str(&policy) {
            response
                .headers_mut()
                .insert(axum::http::header::CONTENT_SECURITY_POLICY, value);
        }
        response.headers_mut().insert(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            axum::http::HeaderValue::from_static("nosniff"),
        );
    }
    response
}

/// Redirects exactly `/demo` to `/demo/`.
///
/// The trailing slash is not cosmetic. Without it the browser resolves the
/// page's relative `<script type="module" src="app.js">` against `/`, requests
/// `/app.js`, and every module 404s — presenting as a blank page whose only
/// symptom is a console error.
///
/// Implemented as a router-level layer rather than a route because `ServeDir` is
/// mounted with `nest_service("/demo", ..)`, which already claims the bare
/// `/demo` path; a competing route there would panic at startup.
///
/// Deliberately temporary rather than permanent: a permanent redirect is cached
/// by the browser and would outlive any change to this mount point.
pub(crate) async fn redirect_bare_demo(request: Request<Body>, next: Next) -> Response {
    if request.uri().path() == "/demo" {
        return Redirect::temporary("/demo/").into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The files that made this a real disclosure rather than tidiness.
    #[test]
    fn the_demo_refuses_to_serve_the_source_tree_it_lives_in() {
        for path in [
            "/demo/ARCHITECTURE-SECURITY-REVIEW.md",
            "/demo/REVIEWER-BRIEF.md",
            "/demo/README.md",
            "/demo/run-demo.sh",
            "/demo/design/notes.md",
            "/demo/scripts/build.py",
        ] {
            assert!(
                !demo_path_is_servable(path),
                "{path} would be published to any visitor of the demo origin"
            );
        }
    }

    /// Test files are `.js` and pass the extension check, so they need their
    /// own rule -- and nothing on the page loads them.
    #[test]
    fn the_demo_refuses_to_serve_its_own_tests() {
        assert!(!demo_path_is_servable("/demo/format.test.js"));
        assert!(!demo_path_is_servable("/demo/dashboard/honesty.test.js"));
    }

    /// The half that matters more: a restriction that breaks the demo is worse
    /// than the disclosure it prevents. These are the paths the page actually
    /// loads, taken from `index.html` and its module graph.
    #[test]
    fn the_demo_still_serves_everything_the_page_loads() {
        for path in [
            "/demo/",
            "/demo/index.html",
            "/demo/app.js",
            "/demo/format.js",
            "/demo/dashboard/scheduling.js",
            "/demo/styles/tokens.css",
            "/demo/styles/shell.css",
            "/demo/styles/panels.css",
            "/demo/telemetry-provenance.js",
            "/demo/assets/icon.svg",
        ] {
            assert!(
                demo_path_is_servable(path),
                "{path} is loaded by the page and must still be served"
            );
        }
    }

    /// The predicate is mounted router-wide, so it must have no opinion about
    /// the API. Without this, one over-broad edit here takes out `/v1`.
    #[test]
    fn the_restriction_has_no_opinion_about_the_api() {
        for path in ["/v1/status", "/v1/debug/kv/blocks", "/metrics", "/demo"] {
            assert!(demo_path_is_servable(path));
        }
    }

    /// Case is not a bypass: `ServeDir` will happily open `NOTES.MD`.
    #[test]
    fn extension_matching_is_case_insensitive() {
        assert!(!demo_path_is_servable("/demo/NOTES.MD"));
        assert!(!demo_path_is_servable("/demo/RUN-DEMO.SH"));
        assert!(demo_path_is_servable("/demo/APP.JS"));
    }

    /// An extensionless file is a source-tree artefact (LICENSE, Makefile,
    /// Dockerfile); nothing the page loads lacks an extension.
    #[test]
    fn extensionless_paths_are_not_served() {
        assert!(!demo_path_is_servable("/demo/LICENSE"));
        assert!(!demo_path_is_servable("/demo/Dockerfile"));
    }

    /// Same host, any port: the demo runs two servers on one host and the page
    /// served by one legitimately polls the other. CSP treats a differing port
    /// as a differing origin, so a bare `'self'` would break the demo.
    #[test]
    fn the_demo_policy_permits_the_sibling_server_on_another_port() {
        let policy = demo_csp_for_host(Some("127.0.0.1:8123"));
        assert!(
            policy.contains("connect-src 'self' http://127.0.0.1:* https://127.0.0.1:*"),
            "got: {policy}"
        );
    }

    /// The policy must follow the host the page was served from, not a
    /// hardcoded loopback literal -- otherwise overriding `BIND_HOST` breaks
    /// the dashboard on stage with a console error as the only symptom.
    #[test]
    fn the_demo_policy_follows_the_host_the_page_was_served_from() {
        assert!(demo_csp_for_host(Some("demo.internal:8124")).contains("http://demo.internal:*"));
        assert!(!demo_csp_for_host(Some("demo.internal:8124")).contains("127.0.0.1"));
    }

    /// The `Host` header is client-controlled. A value carrying a `;` would
    /// append a directive of the attacker's choosing to our own policy.
    #[test]
    fn a_crafted_host_header_cannot_inject_a_directive() {
        for host in [
            "evil.com; script-src *",
            "host with spaces",
            "a\r\nX-Injected: 1",
            "",
            "[::1]:8123",
        ] {
            let policy = demo_csp_for_host(Some(host));
            assert_eq!(
                policy, DEMO_CSP_PREFIX,
                "a host that is not a bare hostname must fall back to 'self' \
                 alone; got: {policy}"
            );
        }
    }

    /// The concessions must stay where they were reasoned about.
    #[test]
    fn the_demo_policy_forbids_inline_script_and_framing() {
        assert!(DEMO_CSP_PREFIX.contains("script-src 'self';"));
        assert!(
            !DEMO_CSP_PREFIX.contains("script-src 'self' 'unsafe-inline'"),
            "the page has no inline script, so this concession is not needed \
             and would forfeit the policy's main protection"
        );
        assert!(DEMO_CSP_PREFIX.contains("frame-ancestors 'none'"));
        assert!(DEMO_CSP_PREFIX.contains("object-src 'none'"));
    }

    /// Measured, not assumed: with only the extension allowlist, `.secret.json`
    /// and `.vscode/settings.json` answered 200 WITH THEIR CONTENTS.
    #[test]
    fn dotfiles_are_refused_even_when_their_extension_is_allowed() {
        for path in [
            "/demo/.secret.json",
            "/demo/.env.json",
            "/demo/.well-known/keys.json",
            "/demo/.vscode/settings.json",
            "/demo/.config/app.js",
        ] {
            assert!(
                !demo_path_is_servable(path),
                "{path} carries an allowlisted extension, so the extension \
                 check alone lets it through"
            );
        }
    }

    /// These were already refused, but only because of their extension. Pinning
    /// them stops a later addition to the allowlist from re-exposing them.
    #[test]
    fn dotfiles_without_an_allowed_extension_stay_refused() {
        for path in ["/demo/.env", "/demo/.npmrc", "/demo/.git/config"] {
            assert!(!demo_path_is_servable(path));
        }
    }

    /// The rule is "segment starts with a dot", not "contains a dot" -- every
    /// real asset has a dot in it.
    #[test]
    fn the_dotfile_rule_does_not_refuse_ordinary_assets() {
        for path in [
            "/demo/index.html",
            "/demo/app.js",
            "/demo/styles/tokens.css",
            "/demo/vendor/chart.min.js",
            "/demo/",
        ] {
            assert!(
                demo_path_is_servable(path),
                "{path} is loaded by the page and must keep working"
            );
        }
    }

    #[test]
    fn missing_directory_resolves_to_none_rather_than_failing() {
        assert!(resolve_demo_assets_dir(Some(PathBuf::from("/nonexistent/demo/dir"))).is_none());
    }

    /// This test used to point at an EMPTY tempdir and assert it was accepted,
    /// which certified the defect: any directory at all could be published on
    /// an unauthenticated origin. It now requires the directory to be able to
    /// serve the dashboard, which is the property that was always meant.
    #[test]
    fn a_directory_holding_the_dashboard_is_used() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("index.html"), "<!doctype html>").unwrap();
        assert_eq!(
            resolve_demo_assets_dir(Some(dir.path().to_path_buf())).as_deref(),
            Some(dir.path())
        );
    }

    /// The case the old test certified as acceptable. `$HOME` and `/etc` are
    /// directories too, and `/demo` needs no credentials.
    #[test]
    fn an_arbitrary_directory_is_not_an_asset_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("taxes.json"), "{}").unwrap();
        std::fs::write(dir.path().join("notes.md"), "private").unwrap();
        assert!(
            resolve_demo_assets_dir(Some(dir.path().to_path_buf())).is_none(),
            "a directory with no entry page cannot serve the dashboard, so \
             mounting it can only publish its contents"
        );
    }

    /// "Exists" is not "can serve": each of these satisfies a bare existence
    /// check and none of them can answer `GET /demo/`.
    #[test]
    fn an_entry_page_that_cannot_be_served_does_not_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("index.html")).unwrap();
        assert!(
            resolve_demo_assets_dir(Some(dir.path().to_path_buf())).is_none(),
            "a DIRECTORY named index.html exists but cannot be served"
        );

        let empty = tempfile::tempdir().expect("tempdir");
        std::fs::write(empty.path().join("index.html"), "").unwrap();
        assert!(
            resolve_demo_assets_dir(Some(empty.path().to_path_buf())).is_none(),
            "a zero-byte entry page exists but renders nothing"
        );

        let dangling = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(
            dangling.path().join("gone.html"),
            dangling.path().join("index.html"),
        )
        .unwrap();
        assert!(
            resolve_demo_assets_dir(Some(dangling.path().to_path_buf())).is_none(),
            "a dangling symlink exists as a link and resolves to nothing"
        );
    }

    /// A symlink to a real page is the normal packaging case and must work.
    #[test]
    fn an_entry_page_reached_through_a_symlink_is_served() {
        let real = tempfile::tempdir().expect("tempdir");
        let page = real.path().join("dashboard.html");
        std::fs::write(&page, "<!doctype html>").unwrap();
        let dir = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(&page, dir.path().join("index.html")).unwrap();
        assert!(resolve_demo_assets_dir(Some(dir.path().to_path_buf())).is_some());
    }

    /// A file is not a directory: pointing `--demo-assets-dir` at `index.html`
    /// must fall through to the actionable message rather than half-working.
    #[test]
    fn a_file_is_not_an_asset_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("index.html");
        std::fs::write(&file, "<!doctype html>").unwrap();
        assert!(resolve_demo_assets_dir(Some(file)).is_none());
    }

    /// The 404 is the first thing anyone hits when they start the server from
    /// the wrong directory, so it has to carry the fix.
    #[test]
    fn missing_assets_message_names_the_flag_and_the_default_path() {
        assert!(MISSING_ASSETS_MESSAGE.contains("--demo-assets-dir"));
        assert!(MISSING_ASSETS_MESSAGE.contains("ONNX_GENAI_DEMO_ASSETS_DIR"));
        assert!(MISSING_ASSETS_MESSAGE.contains(DEFAULT_DEMO_ASSETS_DIR));
    }
}
