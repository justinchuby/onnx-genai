use super::*;
use serde_json::json;
fn settings(root: &Path, timeout: u64, mode: &str) -> ToolsConfig {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_stdio_server.py");
    let document = json!({"mcpServers":{"fixture":{"command":if cfg!(windows){"python"}else{"python3"},"args":[script,"--events",root.join("events"),"--mode",mode,"--stderr-bytes","1048576"],"env":{"NXRT_MCP_FIXTURE":"visible"},"cwd":root,"timeout_ms":timeout}}});
    ToolsConfig {
        mcp: McpSources {
            json: Some(document.to_string().into()),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn configured(root: &Path, timeout: u64, mode: &str) -> Arc<Tools> {
    Tools::new(&settings(root, timeout, mode)).unwrap().unwrap()
}
fn events(root: &Path) -> Vec<Value> {
    std::fs::read_to_string(root.join("events"))
        .unwrap_or_default()
        .split_inclusive('\n')
        .filter(|line| line.ends_with('\n'))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
async fn invoke(tools: &Arc<Tools>, name: &str, params: Value) -> anyhow::Result<ToolValue> {
    let job = Arc::new(tools.admit(params.to_string().len())?);
    let target = tools
        .target(name, job.clone())
        .await?
        .context("fixture tool was not discovered")?;
    job.begin_operation(target.timeout())?;
    target
        .invoke(
            &params.to_string(),
            std::env::current_dir()?,
            false,
            job,
            None,
        )
        .await
}
#[tokio::test]
async fn mcp_stdio_discovers_invokes_normalizes_and_reaps_on_owner_drop() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 3000, "normal");
    let memory = tools.memory.clone();
    let job = Arc::new(tools.admit(0).unwrap());
    let catalog = tools.catalog(job.clone()).await.unwrap();
    assert_eq!(catalog.tools.len(), 7);
    assert_eq!(catalog.tools["fixture_echo"].definition["type"], "mcp");
    drop(job);
    let initial = events(root.path());
    assert_eq!(initial.iter().filter(|v| v.get("pid").is_some()).count(), 1);
    assert!(
        initial
            .iter()
            .any(|v| v["initialize"]["protocolVersion"] == "2024-11-05")
    );
    let result=invoke(&tools,"fixture_echo",json!({"hello":"world","cwd":"model-cannot-override","runtime":"ignored","resp_type":"ignored"})).await.unwrap();
    let text = result.value["plain_text_response"].as_str().unwrap();
    let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(first["arguments"], json!({"hello":"world"}));
    assert_eq!(first["env"], "visible");
    assert_eq!(
        PathBuf::from(first["cwd"].as_str().unwrap())
            .canonicalize()
            .unwrap(),
        root.path().canonicalize().unwrap()
    );
    assert!(text.ends_with("\nsecond"));
    drop(result);
    let result = invoke(&tools, "fixture_fail", json!({})).await.unwrap();
    assert!(
        result.value["error"]
            .as_str()
            .unwrap()
            .ends_with("\nsecond")
    );
    drop(result);
    let result = invoke(&tools, "fixture_ping_peer", json!({}))
        .await
        .unwrap();
    assert!(
        result.value.get("plain_text_response").is_some(),
        "{:?}",
        result.value
    );
    drop(result);
    assert_eq!(
        events(root.path())
            .iter()
            .filter(|v| v.get("pid").is_some())
            .count(),
        2,
        "calls reuse the post-discovery session"
    );
    drop(tools);
    assert_eq!(
        memory.ledger().used(Tier::Host),
        0,
        "joined shutdown releases config/catalog, SDK messages and process credit"
    );
}
#[tokio::test]
async fn mcp_call_timeout_sends_correlated_cancellation_and_releases_admission() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 1000, "normal");
    let failure = tokio::time::timeout(
        Duration::from_secs(5),
        invoke(&tools, "fixture_slow", json!({})),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        failure.value["error"]
            .as_str()
            .is_some_and(|error| error.contains("timeout_ms")),
        "{:?}",
        failure.value
    );
    assert!(
        events(root.path())
            .iter()
            .any(|v| v.get("cancelled").is_some()),
        "stdio cancellation notification must arrive before killing the session: {:?}; events={:?}",
        failure.value,
        events(root.path())
    );
    let observed = events(root.path());
    let id = &observed.iter().find(|v| v["call"] == "slow").unwrap()["id"];
    assert!(observed.iter().any(|v| v.get("cancelled") == Some(id)));
    assert_eq!(tools.slots.available_permits(), 7);
    drop(failure);
    assert_eq!(tools.slots.available_permits(), 8);
    let memory = tools.memory.clone();
    drop(tools);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
}
#[tokio::test]
async fn mcp_unresponsive_initialization_is_bounded_and_does_not_cancel_initialize() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 200, "slow_init");
    let job = Arc::new(tools.admit(0).unwrap());
    let catalog = tokio::time::timeout(Duration::from_secs(5), tools.catalog(job.clone()))
        .await
        .unwrap()
        .unwrap();
    assert!(catalog.tools.is_empty());
    drop(job);
    assert!(
        !events(root.path())
            .iter()
            .any(|v| v.get("cancelled").is_some())
    );
    let memory = tools.memory.clone();
    drop(tools);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
}

#[tokio::test]
async fn rpc_application_error_keeps_the_session_and_releases_its_job() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 3000, "normal");
    let result = invoke(&tools, "fixture_rpc_error", json!({}))
        .await
        .unwrap();
    assert!(
        result.value["error"]
            .as_str()
            .unwrap()
            .contains("RPC_ARGUMENT_ERROR")
    );
    drop(result);
    assert_eq!(tools.slots.available_permits(), 8);
    let result = invoke(&tools, "fixture_echo", json!({"next":true}))
        .await
        .unwrap();
    assert!(
        result.value.get("plain_text_response").is_some(),
        "{}",
        result.value
    );
    drop(result);
    assert_eq!(
        events(root.path())
            .iter()
            .filter(|v| v.get("pid").is_some())
            .count(),
        2
    );
}
#[tokio::test]
async fn discovery_pagination_and_bad_startup_are_real_and_bounded() {
    for mode in ["pagination", "malformed"] {
        let root = tempfile::tempdir().unwrap();
        let tools = configured(root.path(), 3000, mode);
        let result = invoke(&tools, "fixture_echo", json!({"mode":mode}))
            .await
            .unwrap();
        assert!(
            result.value.get("plain_text_response").is_some(),
            "{}",
            result.value
        );
    }
    for (mode, expected) in [
        ("repeated_cursor", "pagination"),
        ("exit_init", "MCP_FIXTURE_BOOT_ERROR"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let tools = configured(root.path(), 3000, mode);
        let job = Arc::new(tools.admit(0).unwrap());
        let error = tools
            .mcp
            .as_ref()
            .unwrap()
            .worker
            .request(0, actor::Operation::Discover, job)
            .await
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains(expected), "{mode}: {error:#}");
    }
}
fn state(config: ToolsConfig) -> crate::AppState {
    crate::AppState::load_from_specs(
        vec![crate::ModelSpec {
            id: "unloaded".into(),
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm"),
            eager: false,
            warmup: false,
        }],
        crate::ServerConfig {
            tools: config,
            ..Default::default()
        },
    )
    .unwrap()
}
async fn http(
    router: axum::Router,
    method: &str,
    body: Value,
    headers: &[(&str, &str)],
) -> axum::response::Response {
    use tower::ServiceExt;
    let mut request = axum::http::Request::builder()
        .method(method)
        .uri("/tools")
        .header("content-type", "application/json");
    for (key, value) in headers {
        request.headers_mut().unwrap().insert(
            axum::http::HeaderName::from_bytes(key.as_bytes()).unwrap(),
            axum::http::HeaderValue::from_str(value).unwrap(),
        );
    }
    router
        .oneshot(
            request
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}
async fn body(response: axum::response::Response) -> Value {
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}
#[tokio::test]
async fn http_mcp_obeys_frontend_headers_and_does_not_load_a_model() {
    use axum::http::StatusCode;
    let root = tempfile::tempdir().unwrap();
    let state = state(settings(root.path(), 3000, "normal"));
    let registry = state.registry.clone();
    let router = crate::app(state);
    let denied = http(
        router.clone(),
        "POST",
        json!({"tool":"fixture_echo"}),
        &[("origin", "https://untrusted.invalid")],
    )
    .await;
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    assert!(events(root.path()).is_empty());
    let discovery = http(router.clone(), "GET", Value::Null, &[]).await;
    assert_eq!(discovery.status(), StatusCode::OK);
    assert_eq!(body(discovery).await.as_array().unwrap().len(), 7);
    let response=http(router.clone(),"POST",json!({"tool":"fixture_echo","params":{"cwd":"model","resp_type":"model","runtime":"model","value":1}}),&[("x-tool-cwd","controller-directory"),("x-resp-type","base64")]).await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = body(response).await;
    let value: Value = serde_json::from_str(
        value["plain_text_response"]
            .as_str()
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        value["arguments"],
        json!({"cwd":"controller-directory","resp_type":"base64","value":1})
    );
    assert_eq!(
        PathBuf::from(value["cwd"].as_str().unwrap())
            .canonicalize()
            .unwrap(),
        root.path().canonicalize().unwrap(),
        "MCP process directory comes from operator configuration"
    );
    let bad = http(
        router.clone(),
        "POST",
        json!({"tool":"fixture_echo","params":false}),
        &[],
    )
    .await;
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    let stream = http(
        router.clone(),
        "POST",
        json!({"tool":"fixture_echo","stream":true}),
        &[],
    )
    .await;
    assert_eq!(stream.status(), StatusCode::NOT_FOUND);
    let runtime = http(
        router,
        "POST",
        json!({"tool":"fixture_echo"}),
        &[("x-tool-runtime", "docker-container:missing")],
    )
    .await;
    assert_eq!(runtime.status(), StatusCode::BAD_REQUEST);
    assert!(registry.lookup_loaded("").unwrap().is_none());
}
#[tokio::test]
async fn mcp_response_owners_outlive_the_node_without_releasing_credit_early() {
    let root = tempfile::tempdir().unwrap();
    let state = state(settings(root.path(), 3000, "normal"));
    let tools = state.tools.as_ref().unwrap().clone();
    let memory = tools.memory.clone();
    let router = crate::app(state);
    let response = http(
        router.clone(),
        "POST",
        json!({"tool":"fixture_echo","params":{"value":"owned"}}),
        &[],
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let alias = bytes.clone();
    drop(bytes);
    drop(router);
    drop(tools);
    assert!(memory.ledger().used(Tier::Host) > 0);
    assert!(
        serde_json::from_slice::<Value>(&alias)
            .unwrap()
            .get("plain_text_response")
            .is_some()
    );
    drop(alias);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
}
#[test]
fn mcp_config_sources_have_explicit_precedence_and_actionable_errors() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("mcp.json");
    std::fs::write(
        &file,
        r#"{"mcpServers":{"same":{"command":"first"},"url":{"url":"https://example.invalid"}}}"#,
    )
    .unwrap();
    let memory = LedgerGovernor::new(LeaseLedger::new(0, 256 * 1024 * 1024, 0));
    let sources=McpSources{file:Some(file),json:Some(r#"{"mcpServers":{"same":{"command":"second"},"other":{"command":"third","env":{"KEY":"secret-marker"},"cwd":"sub","timeout_ms":19}}}"#.into()),..Default::default()};
    assert!(!format!("{sources:?}").contains("secret-marker"));
    let config = Config::load(&sources, &memory, root.path()).unwrap();
    assert_eq!(config.servers.len(), 2);
    assert_eq!(
        config
            .servers
            .iter()
            .find(|s| s.name == "same")
            .unwrap()
            .command,
        "first"
    );
    let other = config.servers.iter().find(|s| s.name == "other").unwrap();
    assert_eq!(other.cwd, root.path().join("sub"));
    assert_eq!(other.timeout, Duration::from_millis(19));
    drop(config);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
    for text in [
        "not-json",
        r#"{"mcpServers":{"broken":{"command":"run","timeout_ms":0}}}"#,
        r#"{"mcpServers":{"broken":{"command":"run","env":{"BAD=NAME":"value"}}}}"#,
    ] {
        let error = Config::load(
            &McpSources {
                json: Some(text.into()),
                ..Default::default()
            },
            &memory,
            root.path(),
        )
        .err()
        .unwrap();
        assert!(
            format!("{error:#}").contains("MCP") || format!("{error:#}").contains("mcpServers")
        );
        assert_eq!(memory.ledger().used(Tier::Host), 0);
    }
    let error = Config::load(
        &McpSources {
            max_message_bytes: std::num::NonZeroUsize::new(8 * 1024 * 1024 + 1).unwrap(),
            ..Default::default()
        },
        &memory,
        root.path(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("8388608"));
}
#[tokio::test]
async fn configured_concurrency_above_eight_is_admitted_on_one_mcp_lane() {
    let root = tempfile::tempdir().unwrap();
    let mut config = settings(root.path(), 10000, "normal");
    config.max_concurrent = std::num::NonZeroUsize::new(16).unwrap();
    let tools = Tools::new(&config).unwrap().unwrap();
    let mut calls = tokio::task::JoinSet::new();
    for index in 0..16 {
        let tools = tools.clone();
        calls.spawn(async move {
            let result = invoke(&tools, "fixture_echo", json!({"index":index}))
                .await
                .unwrap();
            assert!(
                result.value.get("plain_text_response").is_some(),
                "{}",
                result.value
            );
        });
    }
    while let Some(result) = calls.join_next().await {
        result.unwrap();
    }
    assert_eq!(tools.slots.available_permits(), 16);
}

#[tokio::test]
async fn discovery_host_credit_failure_can_retry_without_caching_a_false_empty_catalog() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 3000, "normal");
    let hold = tools.reserve(160 * 1024 * 1024).unwrap();
    let job = Arc::new(tools.admit(0).unwrap());
    let error = tools.catalog(job.clone()).await.err().unwrap();
    assert!(
        error.is::<onnx_runtime_memory_api::MemoryError>(),
        "{error:#}"
    );
    drop(job);
    drop(hold);
    let job = Arc::new(tools.admit(0).unwrap());
    let catalog = tools.catalog(job).await.unwrap();
    assert_eq!(catalog.tools.len(), 7);
}

#[tokio::test]
async fn configurable_message_limit_accepts_a_real_upstream_sized_text_result() {
    let root = tempfile::tempdir().unwrap();
    let mut config = settings(root.path(), 15000, "normal");
    config.mcp.max_message_bytes = std::num::NonZeroUsize::new(8 * 1024 * 1024).unwrap();
    config.max_host_bytes = std::num::NonZeroU64::new(2 * 1024 * 1024 * 1024).unwrap();
    let tools = Tools::new(&config).unwrap().unwrap();
    let size = 8 * 1024 * 1024 - 4096;
    let value = invoke(&tools, "fixture_big", json!({"bytes":size}))
        .await
        .unwrap();
    assert_eq!(
        value.value["plain_text_response"].as_str().unwrap().len(),
        size
    );
}
#[tokio::test]
async fn empty_mcp_configuration_does_not_require_a_large_staging_budget() {
    let tools = Tools::new(&ToolsConfig {
        mcp: McpSources {
            json: Some("{}".into()),
            ..Default::default()
        },
        max_host_bytes: std::num::NonZeroU64::new(1024 * 1024).unwrap(),
        ..Default::default()
    })
    .unwrap();
    assert!(tools.is_none());
}
#[tokio::test]
async fn http_mcp_preflight_credit_refusal_has_503_and_retries_without_spawning() {
    let root = tempfile::tempdir().unwrap();
    let state = state(settings(root.path(), 3000, "normal"));
    let tools = state.tools.as_ref().unwrap().clone();
    let router = crate::app(state);
    let response = http(router.clone(), "GET", Value::Null, &[]).await;
    drop(body(response).await);
    let count = events(root.path())
        .iter()
        .filter(|v| v.get("pid").is_some())
        .count();
    let hold = tools.reserve(254 * 1024 * 1024).unwrap();
    let response = http(router.clone(), "POST", json!({"tool":"fixture_echo"}), &[]).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        body(response).await
    );
    assert_eq!(
        events(root.path())
            .iter()
            .filter(|v| v.get("pid").is_some())
            .count(),
        count
    );
    drop(hold);
    let response = http(router, "POST", json!({"tool":"fixture_echo"}), &[]).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(body(response).await.get("plain_text_response").is_some());
}
#[cfg(unix)]
#[tokio::test]
async fn crashed_mcp_parent_cannot_leave_its_pipe_holding_grandchild_running() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 1000, "normal");
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        invoke(&tools, "fixture_crash", json!({})),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(result.value.get("error").is_some());
    drop(result);
    let records = events(root.path());
    let pid = records
        .iter()
        .find_map(|v| v.get("grandchild").and_then(Value::as_i64))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let status = std::process::Command::new("ps")
                .args(["-o", "stat=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            let status = String::from_utf8_lossy(&status.stdout);
            if status.trim().is_empty() || status.trim().starts_with('Z') {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(tools.slots.available_permits(), 8);
}
#[tokio::test]
async fn tcp_client_disconnect_cancels_an_in_flight_mcp_call_before_its_deadline() {
    use tokio::io::AsyncWriteExt;
    let root = tempfile::tempdir().unwrap();
    let state = state(settings(root.path(), 15000, "normal"));
    let tools = state.tools.as_ref().unwrap().clone();
    let job = Arc::new(tools.admit(0).unwrap());
    tools.catalog(job.clone()).await.unwrap();
    drop(job);
    let baseline = tools.memory.ledger().used(Tier::Host);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, crate::app(state)).await.unwrap();
    });
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    let payload = json!({"tool":"fixture_slow","params":{}}).to_string();
    client.write_all(format!("POST /tools HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",payload.len(),payload).as_bytes()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !events(root.path()).iter().any(|v| v["call"] == "slow") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(client);
    let stopped = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if tools.slots.available_permits() == 8
                && tools.memory.ledger().used(Tier::Host) == baseline
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    server.abort();
    let _ = server.await;
    assert!(
        stopped.is_ok(),
        "TCP disconnect must stop the remote operation before its 15-second deadline"
    );
    assert!(
        events(root.path())
            .iter()
            .any(|v| v.get("cancelled").is_some())
    );
}

#[test]
fn node_shutdown_can_join_children_after_the_callers_tokio_runtime_has_ended() {
    let root = tempfile::tempdir().unwrap();
    let tools = configured(root.path(), 3000, "normal");
    let memory = tools.memory.clone();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = runtime
        .block_on(invoke(&tools, "fixture_echo", json!({"outside":true})))
        .unwrap();
    assert!(result.value.get("plain_text_response").is_some());
    drop(result);
    drop(runtime);
    drop(tools);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
}
#[tokio::test]
async fn discovered_name_collisions_keep_the_first_enabled_capability() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("sample.txt");
    std::fs::write(&path, "from builtin").unwrap();
    for enabled in [false, true] {
        let mut config = settings(root.path(), 3000, "normal");
        let mut document: Value =
            serde_json::from_str(config.mcp.json.as_deref().unwrap()).unwrap();
        let mut entry = document["mcpServers"]["fixture"].take();
        entry["args"]
            .as_array_mut()
            .unwrap()
            .extend([json!("--only-tool"), json!("file")]);
        document["mcpServers"] = json!({"read":entry});
        config.mcp.json = Some(document.to_string().into());
        if enabled {
            config.enabled = vec![BuiltinTool::ReadFile];
        }
        let tools = Tools::new(&config).unwrap().unwrap();
        let result = invoke(&tools, "read_file", json!({"path":path}))
            .await
            .unwrap();
        let text = result.value["plain_text_response"].as_str().unwrap();
        if enabled {
            assert_eq!(text.trim(), "from builtin");
        } else {
            assert!(text.contains("arguments"));
        }
    }
    let mut config = settings(root.path(), 3000, "normal");
    let mut document: Value = serde_json::from_str(config.mcp.json.as_deref().unwrap()).unwrap();
    let mut first = document["mcpServers"]["fixture"].take();
    let mut second = first.clone();
    first["args"]
        .as_array_mut()
        .unwrap()
        .extend([json!("--tool-prefix"), json!("bar_")]);
    first["env"]["NXRT_MCP_FIXTURE"] = json!("one");
    second["env"]["NXRT_MCP_FIXTURE"] = json!("two");
    document["mcpServers"] = json!({"foo":first,"foo_bar":second});
    config.mcp.json = Some(document.to_string().into());
    let tools = Tools::new(&config).unwrap().unwrap();
    let result = invoke(&tools, "foo_bar_echo", json!({})).await.unwrap();
    let first: Value = serde_json::from_str(
        result.value["plain_text_response"]
            .as_str()
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(first["env"], "one");
    assert_eq!(
        tools
            .catalog(Arc::new(tools.admit(0).unwrap()))
            .await
            .unwrap()
            .tools
            .len(),
        7
    );
}

#[tokio::test]
async fn sdk_request_extensions_keep_the_connection_credit_until_the_last_message_owner() {
    use rmcp::transport::Transport as _;
    use tokio::io::AsyncWriteExt;
    let memory = LedgerGovernor::new(LeaseLedger::new(0, 1024 * 1024, 0));
    let connection = Arc::new(actor::ConnectionCredit {
        _lease: memory
            .reserve(
                Tier::Host,
                32 * 1024,
                MemoryRole::Activation,
                HolderId::new(0),
            )
            .unwrap(),
    });
    let (client, mut server) = tokio::io::duplex(2048);
    let (read, write) = tokio::io::split(client);
    let raw =
        rmcp::transport::async_rw::AsyncRwTransport::<rmcp::RoleClient, _, _>::new(read, write)
            .with_max_line_length(1024);
    let mut transport = transport::CreditedTransport::new(
        raw,
        memory.clone(),
        1024,
        Arc::new(std::sync::Mutex::new(None)),
        connection,
    );
    server
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":77,\"method\":\"ping\"}\n")
        .await
        .unwrap();
    let message = transport.receive().await.unwrap();
    drop(transport);
    assert!(memory.ledger().used(Tier::Host) >= 32 * 1024 + 1024 * 32);
    drop(message);
    assert_eq!(memory.ledger().used(Tier::Host), 0);
}
