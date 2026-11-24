use super::*;
use serde_json::json;
fn tools() -> Arc<Tools> {
    Tools::new(&ToolsConfig {
        enabled: vec![BuiltinTool::All],
        ..Default::default()
    })
    .unwrap()
    .unwrap()
}
async fn invoke(
    tools: &Arc<Tools>,
    tool: BuiltinTool,
    params: Value,
    cwd: &Path,
) -> anyhow::Result<ToolValue> {
    tools
        .select_builtin(tool)
        .context("requested builtin is disabled")?
        .invoke(
            &params.to_string(),
            cwd.to_path_buf(),
            false,
            Arc::new(tools.admit(4096)?),
            None,
        )
        .await
}
#[test]
fn explicit_selection_and_origins_define_the_capability() {
    assert!(Tools::new(&ToolsConfig::default()).unwrap().is_none());
    let tools = tools();
    assert_eq!(tools.definitions().as_array().unwrap().len(), 7);
    assert!(tools.origin_allowed("http://localhost:5173"));
    assert!(tools.origin_allowed("https://127.0.0.1:99"));
    assert!(!tools.origin_allowed("http://localhost.evil.test"));
    assert!(!tools.origin_allowed("null"));
    assert!(!tools.origin_allowed("http://user@localhost"));
    assert!(
        Tools::new(&ToolsConfig {
            allow_origins: vec!["https://example.test/path".into()],
            ..Default::default()
        })
        .is_err()
    );
}
#[tokio::test]
async fn filesystem_calls_write_read_base64_and_transactional_edits() {
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    let result = invoke(
        &tools,
        BuiltinTool::WriteFile,
        json!({"path":"sub/a.txt","content":"first\nsecond\nthird\n"}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(result.value["bytes"], 19);
    drop(result);
    let result = invoke(
        &tools,
        BuiltinTool::ReadFile,
        json!({"path":"sub/a.txt","start_line":2,"end_line":2,"append_loc":true}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(result.value["plain_text_response"], "2→second\n");
    drop(result);
    let result=invoke(&tools,BuiltinTool::EditFile,json!({"path":"sub/a.txt","edits":[{"old_text":"first","new_text":"FIRST"},{"old_text":"third","new_text":"THIRD"}]}),root.path()).await.unwrap();
    assert_eq!(result.value["edits_applied"], 2);
    drop(result);
    let before = std::fs::read(root.path().join("sub/a.txt")).unwrap();
    assert!(invoke(&tools,BuiltinTool::EditFile,json!({"path":"sub/a.txt","edits":[{"old_text":"FIRST","new_text":"x"},{"old_text":"FIRST\nsecond","new_text":"y"}]}),root.path()).await.is_err());
    assert_eq!(
        std::fs::read(root.path().join("sub/a.txt")).unwrap(),
        before
    );
    std::fs::write(root.path().join("b.bin"), [0, 255, 7]).unwrap();
    let job = Arc::new(tools.admit(1024).unwrap());
    let result = tools
        .select_builtin(BuiltinTool::ReadFile)
        .unwrap()
        .invoke(
            r#"{"path":"b.bin"}"#,
            root.path().to_owned(),
            true,
            job,
            None,
        )
        .await
        .unwrap();
    assert_eq!(result.value, json!({"base64":"AP8H","size_bytes":3}));
}
#[test]
fn path_resolution_normalizes_components_without_changing_process_cwd() {
    // Exercise through the public worker below; this unit checks path normalization.
    let tools = tools();
    assert_eq!(
        tools.resolve(Path::new("/tmp/project"), "a/../b"),
        PathBuf::from("/tmp/project/b")
    );
}
#[tokio::test]
async fn fuzzy_edit_changes_only_touched_lines() {
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    std::fs::write(
        root.path().join("a"),
        "untouched   \r\nvalue = “a”   \r\nlast   \r\n",
    )
    .unwrap();
    invoke(
        &tools,
        BuiltinTool::EditFile,
        json!({"path":"a","edits":[{"old_text":"value = \"a\"","new_text":"value = \"b\""}]}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(root.path().join("a")).unwrap(),
        "untouched   \r\nvalue = \"b\"\nlast   \r\n"
    );
    std::fs::write(root.path().join("a"), "same  \nsame\n").unwrap();
    assert!(
        invoke(
            &tools,
            BuiltinTool::EditFile,
            json!({"path":"a","edits":[{"old_text":"same  ","new_text":"x"}]}),
            root.path()
        )
        .await
        .is_err()
    );
}
#[tokio::test]
async fn listing_and_grep_respect_git_ignore_and_tracked_exceptions() {
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    assert!(
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root.path())
            .status()
            .unwrap()
            .success()
    );
    for (path, text) in [
        (".gitignore", "*.log\n"),
        ("hidden.log", "hidden"),
        ("tracked.log", "tracked"),
        ("a.rs", "before\nHello[0]\nafter\n"),
    ] {
        std::fs::write(root.path().join(path), text).unwrap();
    }
    assert!(
        std::process::Command::new("git")
            .args(["add", "-f", "tracked.log"])
            .current_dir(root.path())
            .status()
            .unwrap()
            .success()
    );
    let result = invoke(
        &tools,
        BuiltinTool::FileGlobSearch,
        json!({"path":".","include":"*.log"}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(
        result.value["entries"],
        json!([{"path":"tracked.log","type":"file"}])
    );
    drop(result);
    let result=invoke(&tools,BuiltinTool::GrepSearch,json!({"path":".","pattern":"hello[0]","literal":true,"ignore_case":true,"context_lines":1}),root.path()).await.unwrap();
    let text = result.value["plain_text_response"].as_str().unwrap();
    assert!(
        text.contains("a.rs-1-before\n")
            && text.contains("a.rs:2:Hello[0]\n")
            && text.contains("a.rs-3-after\n"),
        "{text}"
    );
}
#[cfg(unix)]
#[tokio::test]
async fn shell_stream_preserves_split_utf8_and_reports_exit_and_timeout() {
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    let result = invoke(
        &tools,
        BuiltinTool::ExecShellCommand,
        json!({"command":"printf hello; printf error >&2; exit 3"}),
        root.path(),
    )
    .await
    .unwrap();
    let text = result.value["plain_text_response"].as_str().unwrap();
    assert!(text.contains("hello") && text.contains("error") && text.contains("[exit code: 3]"));
    drop(result);
    let (sender, mut receiver) = tokio::sync::mpsc::channel(16);
    let job = Arc::new(tools.admit(1024).unwrap());
    let result = tools
        .select_builtin(BuiltinTool::ExecShellCommand)
        .unwrap()
        .invoke(
            r#"{"command":"printf '\\342'; sleep 0.05; printf '\\202\\254'"}"#,
            root.path().to_owned(),
            false,
            job,
            Some(sender),
        )
        .await
        .unwrap();
    assert!(result.value.is_null());
    let mut chunks = String::new();
    while let Some(value) = receiver.recv().await {
        chunks.push_str(value["chunk"].as_str().unwrap());
    }
    assert_eq!(chunks, "€\n[exit code: 0]");
    drop(result);
    let result = invoke(
        &tools,
        BuiltinTool::ExecShellCommand,
        json!({"command":"sleep 10","timeout":1}),
        root.path(),
    )
    .await
    .unwrap();
    assert!(
        result.value["plain_text_response"]
            .as_str()
            .unwrap()
            .contains("exit due to timed out")
    );
}
#[test]
fn admission_and_body_credit_are_bounded() {
    let tools = Tools::new(&ToolsConfig {
        enabled: vec![BuiltinTool::All],
        max_concurrent: std::num::NonZeroUsize::new(1).unwrap(),
        ..Default::default()
    })
    .unwrap()
    .unwrap();
    let job = tools.admit(0).unwrap();
    assert!(tools.admit(0).is_err());
    assert!(tools.memory.ledger().used(Tier::Host) > 0);
    drop(job);
    assert_eq!(tools.memory.ledger().used(Tier::Host), 0);
    assert!(tools.admit(0).is_ok());
}

fn app_state(enabled: bool) -> crate::AppState {
    crate::AppState::load_from_specs(
        vec![crate::ModelSpec {
            id: "unloaded".into(),
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm"),
            eager: false,
            warmup: false,
        }],
        crate::ServerConfig {
            tools: ToolsConfig {
                enabled: if enabled {
                    vec![BuiltinTool::All]
                } else {
                    Vec::new()
                },
                max_concurrent: std::num::NonZeroUsize::new(1).unwrap(),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}
async fn http(
    state: crate::AppState,
    method: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> axum::response::Response {
    use tower::ServiceExt;
    let body = body.to_string();
    let mut request = axum::http::Request::builder()
        .method(method)
        .uri("/tools")
        .header("content-type", "application/json")
        .header("content-length", body.len());
    for (key, value) in headers {
        request.headers_mut().unwrap().insert(
            axum::http::HeaderName::from_bytes(key.as_bytes()).unwrap(),
            axum::http::HeaderValue::from_str(value).unwrap(),
        );
    }
    crate::app(state)
        .oneshot(request.body(axum::body::Body::from(body)).unwrap())
        .await
        .unwrap()
}
async fn json_body(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
#[tokio::test]
async fn http_tools_are_opt_in_and_discovery_never_loads_a_model() {
    let disabled = http(app_state(false), "GET", Value::Null, &[]).await;
    assert_eq!(disabled.status(), axum::http::StatusCode::FORBIDDEN);
    let state = app_state(true);
    let registry = state.registry.clone();
    let response = http(state, "GET", Value::Null, &[]).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(json_body(response).await.as_array().unwrap().len(), 7);
    assert!(registry.lookup_loaded("").unwrap().is_none());
}
#[tokio::test]
async fn http_headers_control_cwd_and_base64_without_model_supplied_overrides() {
    let root = tempfile::tempdir().unwrap();
    let state = app_state(true);
    let cwd = root.path().to_str().unwrap();
    let response=http(state.clone(),"POST",json!({"tool":"write_file","params":{"path":"file","content":"hello","cwd":"/missing","runtime":"ignored"}}),&[("x-tool-cwd",cwd)]).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(json_body(response).await["bytes"], 5);
    let response = http(
        state.clone(),
        "POST",
        json!({"tool":"read_file","params":{"path":"file","resp_type":"plain"}}),
        &[("x-tool-cwd", cwd), ("x-resp-type", "base64")],
    )
    .await;
    assert_eq!(
        json_body(response).await,
        json!({"base64":"aGVsbG8=","size_bytes":5})
    );
    let response = http(
        state.clone(),
        "POST",
        json!({"tool":"write_file","params":{"path":"bad","content":"no"}}),
        &[
            ("x-tool-cwd", cwd),
            ("x-tool-runtime", "docker-container:unavailable"),
        ],
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    assert!(!root.path().join("bad").exists());
    let response = http(state, "POST", json!({"tool":"missing","params":{}}), &[]).await;
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}
#[tokio::test]
async fn origin_and_content_type_checks_precede_side_effects_and_cors_is_route_local() {
    use tower::ServiceExt;
    let root = tempfile::tempdir().unwrap();
    let state = app_state(true);
    let cwd = root.path().to_str().unwrap();
    let body = json!({"tool":"write_file","params":{"path":"bad","content":"no"}});
    let response = http(
        state.clone(),
        "POST",
        body.clone(),
        &[("x-tool-cwd", cwd), ("origin", "https://untrusted.invalid")],
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::FORBIDDEN);
    assert!(!root.path().join("bad").exists());
    let encoded = body.to_string();
    let response = crate::app(state.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/tools")
                .header("content-type", "text/plain")
                .header("content-length", encoded.len())
                .header("x-tool-cwd", cwd)
                .body(axum::body::Body::from(encoded))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
    assert!(!root.path().join("bad").exists());
    for (origin, allowed) in [
        ("http://localhost:5173", true),
        ("https://untrusted.invalid", false),
    ] {
        let response = crate::app(state.clone())
            .oneshot(
                axum::http::Request::builder()
                    .method("OPTIONS")
                    .uri("/tools")
                    .header("origin", origin)
                    .header("access-control-request-method", "POST")
                    .header("access-control-request-headers", "content-type,x-tool-cwd")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .is_some(),
            allowed
        );
    }
    let response = crate::app(state)
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .header("origin", "http://localhost:5173")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        !response
            .headers()
            .contains_key("access-control-allow-origin")
    );
}
#[tokio::test]
async fn http_response_bytes_own_admission_and_memory_until_last_clone_drops() {
    let state = app_state(true);
    let tools = state.tools.clone().unwrap();
    let response = http(state.clone(), "GET", Value::Null, &[]).await;
    assert!(tools.memory.ledger().used(Tier::Host) > 0);
    let blocked = http(state, "POST", json!({"tool":"get_info"}), &[]).await;
    assert_eq!(
        blocked.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    drop(blocked);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let alias = bytes.clone();
    drop(bytes);
    assert!(tools.memory.ledger().used(Tier::Host) > 0);
    drop(alias);
    assert_eq!(tools.memory.ledger().used(Tier::Host), 0);
    assert_eq!(tools.slots.available_permits(), 1);
}
#[cfg(unix)]
#[tokio::test]
async fn http_shell_stream_uses_upstream_chunk_and_done_events() {
    let state = app_state(true);
    let response = http(
        state,
        "POST",
        json!({"tool":"exec_shell_command","params":{"command":"printf hello"},"stream":true}),
        &[],
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(
        response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    let values = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values.last(), Some(&json!({"done":true})));
    assert!(
        values
            .iter()
            .any(|v| v["chunk"].as_str().is_some_and(|s| s.contains("hello")))
    );
    assert!(!text.contains("[DONE]"));
}
#[cfg(unix)]
#[tokio::test]
async fn disconnect_cancels_the_process_group_even_after_the_shell_exits() {
    let root = tempfile::tempdir().unwrap();
    let state = app_state(true);
    let tools = state.tools.clone().unwrap();
    let response=http(state,"POST",json!({"tool":"exec_shell_command","params":{"command":"sleep 30 & echo $! > child.pid"},"stream":true}),&[("x-tool-cwd",root.path().to_str().unwrap())]).await;
    let pid: i32 = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(text) = std::fs::read_to_string(root.path().join("child.pid"))
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if tools.slots.available_permits() == 1 && tools.memory.ledger().used(Tier::Host) == 0 {
                let status = std::process::Command::new("ps")
                    .args(["-o", "stat=", "-p", &pid.to_string()])
                    .output()
                    .unwrap();
                let status = String::from_utf8_lossy(&status.stdout);
                if status.trim().is_empty() || status.trim().starts_with('Z') {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn readonly_git_listing_never_executes_a_repository_fsmonitor_hook() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    assert!(
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root.path())
            .status()
            .unwrap()
            .success()
    );
    std::fs::write(root.path().join("visible"), "text").unwrap();
    assert!(
        std::process::Command::new("git")
            .args(["add", "visible"])
            .current_dir(root.path())
            .status()
            .unwrap()
            .success()
    );
    let script = root.path().join("monitor");
    std::fs::write(
        &script,
        "#!/bin/sh\nprintf called > monitor-called\nprintf '\\0'\n",
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        std::process::Command::new("git")
            .args(["config", "core.fsmonitor", script.to_str().unwrap()])
            .current_dir(root.path())
            .status()
            .unwrap()
            .success()
    );
    assert!(
        std::process::Command::new("git")
            .args(["ls-files", "--cached", "--others", "--exclude-standard"])
            .current_dir(root.path())
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        root.path().join("monitor-called").exists(),
        "control did not invoke the fixture hook"
    );
    std::fs::remove_file(root.path().join("monitor-called")).unwrap();
    invoke(
        &tools,
        BuiltinTool::FileGlobSearch,
        json!({"path":"."}),
        root.path(),
    )
    .await
    .unwrap();
    assert!(!root.path().join("monitor-called").exists());
}
#[tokio::test]
async fn search_limits_directory_depth_backreferences_and_read_ranges_are_real() {
    let root = tempfile::tempdir().unwrap();
    let tools = tools();
    std::fs::create_dir_all(root.path().join("nested/deep")).unwrap();
    std::fs::write(root.path().join("nested/deep/a"), "abcabc\n").unwrap();
    let result = invoke(
        &tools,
        BuiltinTool::GrepSearch,
        json!({"path":".","pattern":r"(abc)\1"}),
        root.path(),
    )
    .await
    .unwrap();
    assert!(
        result.value["plain_text_response"]
            .as_str()
            .unwrap()
            .contains("nested/deep/a:abcabc")
    );
    drop(result);
    for i in 0..105 {
        std::fs::write(root.path().join(format!("file-{i}")), "x").unwrap();
    }
    let result = invoke(
        &tools,
        BuiltinTool::FileGlobSearch,
        json!({"path":".","limit":1000,"max_depth":1}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(result.value["entries"].as_array().unwrap().len(), 100);
    let text = result.value["plain_text_response"].as_str().unwrap();
    assert!(text.contains("Total matches: 105"));
    assert!(!text.contains("nested/deep"));
    drop(result);
    std::fs::write(
        root.path().join("large"),
        format!("{}\nselected\n", "a".repeat(TEXT_BYTES + 1)),
    )
    .unwrap();
    assert!(
        invoke(
            &tools,
            BuiltinTool::ReadFile,
            json!({"path":"large"}),
            root.path()
        )
        .await
        .is_err()
    );
    let result = invoke(
        &tools,
        BuiltinTool::ReadFile,
        json!({"path":"large","start_line":2,"end_line":2}),
        root.path(),
    )
    .await
    .unwrap();
    assert_eq!(result.value["plain_text_response"], "selected\n");
}
#[cfg(unix)]
#[tokio::test]
async fn regular_file_checks_do_not_block_on_named_pipes() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        std::process::Command::new("mkfifo")
            .arg(root.path().join("pipe"))
            .status()
            .unwrap()
            .success()
    );
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        invoke(
            &tools(),
            BuiltinTool::ReadFile,
            json!({"path":"pipe"}),
            root.path(),
        ),
    )
    .await
    .unwrap();
    assert!(
        result
            .err()
            .expect("a pipe must be refused")
            .to_string()
            .contains("regular file")
    );
}
#[tokio::test]
async fn trusted_remote_origins_are_explicit_and_empty_budgets_fail_before_io() {
    use tower::ServiceExt;
    let mut state = app_state(true);
    let config = ToolsConfig {
        enabled: vec![BuiltinTool::ReadFile],
        allow_origins: vec!["https://trusted.invalid".into()],
        ..Default::default()
    };
    state.tools = Tools::new(&config).unwrap();
    let response = crate::app(state)
        .oneshot(
            axum::http::Request::builder()
                .uri("/tools")
                .header("origin", "https://trusted.invalid")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(
        response.headers()["access-control-allow-origin"],
        "https://trusted.invalid"
    );
    assert!(
        Tools::new(&ToolsConfig {
            enabled: vec![BuiltinTool::All],
            max_host_bytes: std::num::NonZeroU64::new(1).unwrap(),
            ..Default::default()
        })
        .is_err()
    );
    let tools = Tools::new(&ToolsConfig {
        enabled: vec![BuiltinTool::All],
        max_host_bytes: std::num::NonZeroU64::new(300 * 1024).unwrap(),
        ..Default::default()
    })
    .unwrap()
    .unwrap();
    assert!(tools.admit(64 * 1024).is_err());
    assert_eq!(tools.memory.ledger().used(Tier::Host), 0);
    assert_eq!(tools.slots.available_permits(), 8);
}

#[tokio::test]
async fn chunked_requests_release_unused_body_credit_before_file_search() {
    use tower::ServiceExt;
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("found"), "yes").unwrap();
    let state = app_state(true);
    let response = crate::app(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/tools")
                .header("content-type", "application/json")
                .header("x-tool-cwd", root.path().to_str().unwrap())
                .body(axum::body::Body::from(
                    json!({"tool":"file_glob_search","params":{"path":"."}}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(
        json_body(response).await["entries"],
        json!([{"path":"found","type":"file"}])
    );
}
#[tokio::test]
async fn nonregular_regex_work_is_refused_before_unbounded_matching() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a"), "a").unwrap();
    let result = invoke(
        &tools(),
        BuiltinTool::GrepSearch,
        json!({"path":"a","pattern":r"(a?)\1{0,1000000}"}),
        root.path(),
    )
    .await;
    let error = result
        .err()
        .expect("work limit must reject this expression");
    assert!(format!("{error:#}").contains("work budget"), "{error:#}");
}

#[tokio::test]
async fn body_errors_distinguish_size_limits_from_incomplete_transport() {
    use tower::ServiceExt;
    let state = app_state(true);
    let oversized = crate::app(state.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/tools")
                .header("content-type", "application/json")
                .header("content-length", "1")
                .body(axum::body::Body::from(r#"{"tool":"get_info"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        oversized.status(),
        axum::http::StatusCode::PAYLOAD_TOO_LARGE
    );
    let stream = tokio_stream::iter([Err::<axum::body::Bytes, _>(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "fixture transport ended early",
    ))]);
    let incomplete = crate::app(state.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/tools")
                .header("content-type", "application/json")
                .body(axum::body::Body::from_stream(stream))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(incomplete.status(), axum::http::StatusCode::BAD_REQUEST);
    assert!(
        json_body(incomplete).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("fixture transport ended early")
    );
    let response = http(
        state,
        "POST",
        json!({"tool":"get_info"}),
        &[("content-type", "Application/JSON; charset=utf-8")],
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}
