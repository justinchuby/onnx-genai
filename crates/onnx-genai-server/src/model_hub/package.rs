//! Selective installation follows the authored package index, not weight names.
use super::*;
use onnx_model_package::{Component, ComponentReference, Manifest};
use serde::de::DeserializeOwned;
use std::path::Component as PathComponent;

fn index_path(manifest: &Manifest, reference: &str) -> anyhow::Result<String> {
    let path = manifest.portable_reference(reference)?;
    let value = path
        .components()
        .map(|part| match part {
            PathComponent::Normal(value) => {
                value.to_str().context("package index path must be UTF-8")
            }
            _ => {
                anyhow::bail!("package index path {reference:?} is not a normalized relative path")
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .join("/");
    ensure!(
        value.is_empty() || valid_file(&value),
        "package index reference {reference:?} is not a portable cache path; publish regular package-relative artifacts"
    );
    Ok(value)
}

impl HubClient {
    async fn read_package_index<T: DeserializeOwned>(
        &self,
        model: &HubModel,
        commit: &str,
        file: &RepositoryFile,
    ) -> anyhow::Result<T> {
        let size = file
            .size
            .with_context(|| format!("package index {:?} omits its file size", file.rfilename))?;
        ensure!(
            size <= onnx_model_package::MAX_JSON_BYTES,
            "package index {:?} exceeds the {}-byte JSON limit; reduce the index",
            file.rfilename,
            onnx_model_package::MAX_JSON_BYTES
        );
        let sha256 = admit_lfs(&file.rfilename, size, file.lfs.as_ref())?;
        let mut response = self
            .client
            .get(self.immutable_file_url(model, commit, &file.rfilename))
            .send()
            .await
            .with_context(|| {
                format!(
                    "fetch package index {:?} for {} at immutable commit {commit}",
                    file.rfilename,
                    model.id()
                )
            })?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Hub refused package index {:?}; restore its declared file",
                    file.rfilename
                )
            })?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("read immutable package index")?
        {
            ensure!(
                chunk.len() as u64 <= size.saturating_sub(bytes.len() as u64),
                "package index {:?} exceeds its declared {size}-byte size; repair Hub metadata",
                file.rfilename
            );
            bytes.extend_from_slice(&chunk);
        }
        ensure!(
            bytes.len() as u64 == size,
            "package index {:?} ended before its declared {size}-byte size; retry or repair the repository",
            file.rfilename
        );
        if let Some(expected) = sha256 {
            ensure!(
                format!("{:x}", Sha256::digest(&bytes)) == expected,
                "package index {:?} failed its LFS SHA-256 check; restore the immutable repository file",
                file.rfilename
            );
        }
        serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "parse package index {:?} for model {}; repair the JSON document",
                file.rfilename,
                model.id()
            )
        })
    }

    pub(super) async fn select_package_files(
        &self,
        model: &HubModel,
        info: &mut RepositoryInfo,
    ) -> anyhow::Result<()> {
        let Some(_) = model.variant() else {
            return Ok(());
        };
        validate_commit(&info.sha, model.id())?;
        let mut names = BTreeSet::new();
        for file in &info.siblings {
            ensure!(
                valid_file(&file.rfilename) && names.insert(file.rfilename.as_str()),
                "Hub file {:?} has an unsafe, reserved or duplicate relative path; publish regular files beneath the package root",
                file.rfilename
            );
        }
        let file = info.siblings.iter().find(|file| file.rfilename == "manifest.json")
            .with_context(|| format!("model {} requires a model-package manifest.json for its explicit variant; publish the executable graph and variant index", model.id()))?;
        let manifest: Manifest = self.read_package_index(model, &info.sha, file).await?;
        let component_name = manifest.model_component_name()?.to_owned();
        let mut indexes = BTreeSet::from([String::from("manifest.json")]);
        let component = match &manifest.components[&component_name] {
            ComponentReference::Inline(component) => component.as_ref().clone(),
            ComponentReference::External(reference) => {
                let path = index_path(&manifest, reference)?;
                let path = if names.contains(path.as_str()) {
                    path
                } else if path.is_empty() {
                    "component.json".into()
                } else {
                    format!("{path}/component.json")
                };
                let file = info.siblings.iter().find(|file| file.rfilename == path)
                    .with_context(|| format!("component {component_name:?} references missing package index {path:?}; publish its component.json"))?;
                indexes.insert(path);
                self.read_package_index::<Component>(model, &info.sha, file)
                    .await?
            }
        };
        component.validate_index(&component_name)?;
        let (_, selected) = component.select_variant(&model.package_selection())?;
        let directory = index_path(&manifest, &selected.variant_directory)?;
        ensure!(
            directory.is_empty()
                || info
                    .siblings
                    .iter()
                    .any(|file| file.rfilename.starts_with(&format!("{directory}/"))),
            "selected variant {:?} directory {:?} contains no repository artifacts; publish its executable package files",
            model.variant(),
            selected.variant_directory
        );
        // Shared assets are common package payloads. Preserve their authored
        // roots and the format's standard shared-assets directory; exclude
        // unrelated variant directories and repository metadata.
        let mut roots = BTreeSet::from([directory, String::from("shared_assets")]);
        for key in manifest.shared_assets.keys() {
            let reference = if key.starts_with("sha256:") {
                key.clone()
            } else {
                format!("sha256:{key}")
            };
            roots.insert(index_path(&manifest, &reference)?);
        }
        info.siblings.retain(|file| {
            indexes.contains(&file.rfilename)
                || roots.contains("")
                || roots.contains(&file.rfilename)
                || file
                    .rfilename
                    .match_indices('/')
                    .any(|(offset, _)| roots.contains(&file.rfilename[..offset]))
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    pub(crate) async fn variant_hub(
        fixture_name: &str,
        external: bool,
        shared: bool,
        fault: Option<&str>,
    ) -> (
        HubClient,
        Arc<Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::{Router, extract::Request, response::IntoResponse, routing::get};
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(fixture_name);
        let mut files = BTreeMap::new();
        for variant in ["large", "small"] {
            for name in [
                "model.onnx.textproto",
                "inference_metadata.yaml",
                "tokenizer.json",
            ] {
                files.insert(
                    format!("{variant}/{name}"),
                    std::fs::read(fixture.join(name)).unwrap(),
                );
            }
        }
        let tokenizer_reference = if shared {
            format!("sha256:{}", "b".repeat(64))
        } else {
            ".".into()
        };
        if shared {
            let tokenizer = files.remove("small/tokenizer.json").unwrap();
            files.remove("large/tokenizer.json");
            let root = format!("shared_assets/sha256-{}", "b".repeat(64));
            files.insert(format!("{root}/tokenizer.json"), tokenizer);
            for variant in ["large", "small"] {
                let file = format!("{variant}/inference_metadata.yaml");
                let mut metadata: serde_json::Value =
                    serde_yaml::from_slice(&files[&file]).unwrap();
                if let Some(artifacts) = metadata
                    .get_mut("package")
                    .and_then(|package| package.get_mut("tokenizer"))
                    .and_then(|tokenizer| tokenizer.get_mut("artifacts"))
                    .and_then(serde_json::Value::as_array_mut)
                {
                    assert_eq!(artifacts.len(), 1);
                    artifacts[0]["location"] =
                        serde_json::json!(format!("{tokenizer_reference}/tokenizer.json"));
                    files.insert(file, serde_yaml::to_string(&metadata).unwrap().into_bytes());
                }
            }
            files.insert(
                format!("{root}/tokenizer_config.json"),
                br#"{"chat_template":"{{ messages[0].content }}"}"#.to_vec(),
            );
        }
        let entry = |directory: &str| {
            serde_json::json!({"variant_directory":directory,"executor_info":{"nxrt":{
                "model_file":"model.onnx.textproto","inference_metadata":"inference_metadata.yaml","tokenizer":tokenizer_reference
            }}})
        };
        let component = serde_json::json!({"component_name":"model","variants":{"large":entry("large"),"other":entry("small"),"small":entry("small")}});
        let reference = if external {
            files.insert(
                "components/component.json".into(),
                serde_json::to_vec(&component).unwrap(),
            );
            serde_json::json!("components")
        } else {
            component
        };
        let mut manifest =
            serde_json::json!({"schema_version":"1.0","components":{"model":reference}});
        match fault {
            Some("version") => manifest["schema_version"] = serde_json::json!("2.0"),
            Some("directory") => {
                manifest["components"]["model"]["variants"]["small"]["variant_directory"] =
                    serde_json::json!("../outside")
            }
            Some("component") => {
                manifest["components"]["model"]["component_name"] = serde_json::json!("other")
            }
            Some("component_path") => {
                manifest["components"]["model"] = serde_json::json!("../component")
            }
            Some("shared_path") => {
                manifest["shared_assets"] = serde_json::json!({"b".repeat(64):"../outside"})
            }
            _ => {}
        }
        files.insert(
            "manifest.json".into(),
            serde_json::to_vec(&manifest).unwrap(),
        );
        if fault == Some("json") {
            files.insert("manifest.json".into(), b"{".to_vec());
        }
        let mut siblings = files
            .iter()
            .map(|(name, bytes)| {
                serde_json::json!({"rfilename":name,"size":bytes.len(),"lfs":{
                    "size":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))
                }})
            })
            .collect::<Vec<_>>();
        // Repository size exceeds the selected transfer quota. No body is
        // available for this unselected weight, so any accidental fetch fails.
        siblings.push(
            serde_json::json!({"rfilename":"large/unselected.gguf","size":1_000_000_000_u64}),
        );
        let index = siblings
            .iter_mut()
            .find(|file| file["rfilename"] == "manifest.json")
            .unwrap();
        match fault {
            Some("short") | Some("long") | Some("oversize") => {
                let original = index["size"].as_u64().unwrap();
                let size = match fault {
                    Some("short") => original + 1,
                    Some("long") => original - 1,
                    _ => onnx_model_package::MAX_JSON_BYTES + 1,
                };
                index["size"] = serde_json::json!(size);
                index["lfs"]["size"] = serde_json::json!(size);
            }
            Some("hash") => index["lfs"]["sha256"] = serde_json::json!("0".repeat(64)),
            Some("lfs") => index["lfs"]["sha256"] = serde_json::json!("bad"),
            _ => {}
        }
        if fault == Some("unsafe_sibling") {
            siblings.push(serde_json::json!({"rfilename":"../unsafe","size":1}));
        }
        let info =
            serde_json::to_vec(&serde_json::json!({"sha":"a".repeat(40),"siblings":siblings}))
                .unwrap();
        let trace = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&trace);
        let router = Router::new().fallback(get(move |request: Request| {
            let trace = Arc::clone(&captured);
            let files = files.clone();
            let info = info.clone();
            async move {
                let path = request.uri().path().to_owned();
                trace.lock().unwrap().push(path.clone());
                if path == "/api/models/owner/package/revision/main" {
                    return info.into_response();
                }
                let prefix = format!("/owner/package/resolve/{}/", "a".repeat(40));
                match path.strip_prefix(&prefix).and_then(|name| files.get(name)) {
                    Some(bytes) => bytes.clone().into_response(),
                    None => axum::http::StatusCode::NOT_FOUND.into_response(),
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint =
            reqwest::Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (
            HubClient {
                client: reqwest::Client::new(),
                endpoint,
            },
            trace,
            task,
        )
    }

    #[tokio::test]
    async fn selected_package_plan_downloads_only_the_authored_variant_and_keeps_its_identity() {
        for (external, shared) in [(false, false), (true, true)] {
            let (client, trace, task) =
                variant_hub("tiny-native-engine", external, shared, None).await;
            let model = HubModel::new("owner/package:small".into(), "main".into()).unwrap();
            assert_eq!(model.repository, "owner/package");
            let limits = HubLimits {
                bytes: 64 * 1024,
                files: 16,
            };
            let plan = client.plan(model, limits).await.unwrap();
            assert_eq!(plan.id(), "owner/package:small");
            assert_eq!(plan.file_count(), if shared { 6 } else { 4 });
            assert!(plan.files().all(|(name, _)| name == "manifest.json"
                || name.starts_with("small/")
                || name.starts_with("components/")
                || name.starts_with("shared_assets/")));
            let plan = DownloadPlan::decode(&plan.encode().unwrap(), limits).unwrap();
            assert_eq!(plan.model().variant(), Some("small"));
            let root = tempfile::tempdir().unwrap();
            let (cache, _, _) = crate::model_cache::ModelCache::open(
                root.path(),
                crate::model_cache::CacheLimits {
                    bytes: 1024 * 1024,
                    files: 32,
                    models: 2,
                },
            )
            .unwrap();
            let mut transfer = cache.begin(plan).unwrap();
            let (_control, mut pause) = tokio::sync::watch::channel(false);
            assert_eq!(
                client
                    .download(&mut transfer, &mut pause, |_| {})
                    .await
                    .unwrap(),
                DownloadStatus::Complete
            );
            let selected = onnx_genai_ort::ModelDirectory::load_with_package_selection(
                transfer.path(),
                &transfer.plan().model().package_selection(),
            )
            .unwrap();
            assert!(selected.model_path.ends_with("small/model.onnx.textproto"));
            assert!(!transfer.path().join("large").exists());
            {
                let seen = trace.lock().unwrap();
                assert!(
                    seen.iter()
                        .all(|path| !path.contains(":small") && !path.contains("/large/")),
                    "{seen:?}"
                );
                assert!(
                    seen.iter()
                        .filter(|path| path.ends_with("/manifest.json"))
                        .count()
                        >= 2
                );
            }
            let outcome = transfer.run(&client, &mut pause, |_| {}).await;
            let crate::model_cache::TransferOutcome::Complete(completed) = outcome else {
                panic!("completed selected transfer must resume and publish");
            };
            let published = completed.publish().unwrap();
            let spec = published.spec();
            assert_eq!(spec.package_variant.as_deref(), Some("small"));
            assert_eq!(spec.id, "owner/package:small");
            #[cfg(feature = "native-backend")]
            {
                for backend in [
                    onnx_genai_engine::EngineDecodeBackend::Native,
                    onnx_genai_engine::EngineDecodeBackend::Ort,
                ] {
                    let mut engine = onnx_genai_engine::Engine::from_dir(
                        &spec.path,
                        onnx_genai_engine::EngineConfig {
                            decode_backend: backend,
                            package_variant: spec.package_variant.clone(),
                            native_device: Some(onnx_genai_engine::NativeDecodeDevice::Cpu),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let mut request = onnx_genai_engine::GenerateRequest::new(
                        onnx_genai_engine::GeneratePrompt::TokenIds(vec![0]),
                    );
                    request.options.greedy = true;
                    request.options.max_new_tokens = 2;
                    request.options.stop_on_eos = false;
                    assert_eq!(
                        engine.generate(request).unwrap().token_ids,
                        vec![1, 1],
                        "{backend:?}"
                    );
                }
                let mut spec = spec;
                spec.eager = true;
                let state =
                    crate::AppState::load_from_specs(vec![spec], crate::ServerConfig::default())
                        .unwrap();
                if shared {
                    assert!(
                        state
                            .registry
                            .lookup_loaded("owner/package:small")
                            .unwrap()
                            .unwrap()
                            .chat_template
                            .is_some()
                    );
                }
                drop(state);
            }
            drop(published);
            drop(cache);
            let (_reopened, packages, pending) = crate::model_cache::ModelCache::open(
                root.path(),
                crate::model_cache::CacheLimits {
                    bytes: 1024 * 1024,
                    files: 32,
                    models: 2,
                },
            )
            .unwrap();
            assert!(pending.is_empty());
            assert_eq!(packages.len(), 1);
            assert_eq!(packages[0].spec().package_variant.as_deref(), Some("small"));
            assert_eq!(packages[0].id(), "owner/package:small");
            task.abort();
        }
    }

    #[cfg(feature = "native-backend")]
    #[tokio::test]
    async fn selected_package_workflow_runs_through_the_real_systemone_route() {
        use tower::ServiceExt;
        for shared in [false, true] {
            let (client, _, task) =
                variant_hub("onnx_genai_workflows/decision", true, shared, None).await;
            let plan = client
                .plan(
                    HubModel::new("owner/package:small".into(), "main".into()).unwrap(),
                    HubLimits {
                        bytes: 1024 * 1024,
                        files: 16,
                    },
                )
                .await
                .unwrap();
            let root = tempfile::tempdir().unwrap();
            let (cache, _, _) = crate::model_cache::ModelCache::open(
                root.path(),
                crate::model_cache::CacheLimits {
                    bytes: 4 * 1024 * 1024,
                    files: 32,
                    models: 2,
                },
            )
            .unwrap();
            let transfer = cache.begin(plan).unwrap();
            let (_control, mut pause) = tokio::sync::watch::channel(false);
            let crate::model_cache::TransferOutcome::Complete(completed) =
                transfer.run(&client, &mut pause, |_| {}).await
            else {
                panic!("selected composed package transfer must complete");
            };
            let published = completed.publish().unwrap();
            assert!(!published.spec().path.join("large").exists());
            for backend in [
                onnx_genai_engine::EngineDecodeBackend::Native,
                onnx_genai_engine::EngineDecodeBackend::Ort,
            ] {
                let mut spec = published.spec();
                spec.eager = true;
                let mut config = crate::ServerConfig::default();
                config.engine_config.decode_backend = backend;
                config.engine_config.native_device =
                    Some(onnx_genai_engine::NativeDecodeDevice::Cpu);
                let state = crate::AppState::load_from_specs(vec![spec], config).unwrap();
                let router = crate::app(state);
                for (text, expected) in [("hello", "a"), ("world", "b")] {
                    let body = serde_json::json!({"model":"owner/package:small","state":text,"questions":{
                        "select":{"type":"choice","instructions":"the","criteria":{"a":"hello","b":"world"}}
                    }});
                    let response = router
                        .clone()
                        .oneshot(
                            axum::http::Request::post("/v1/systemone")
                                .header("content-type", "application/json")
                                .body(axum::body::Body::from(body.to_string()))
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    let status = response.status();
                    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                        .await
                        .unwrap();
                    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(status, axum::http::StatusCode::OK, "{backend:?}: {value}");
                    assert_eq!(value["model"], "owner/package:small");
                    assert_eq!(
                        value["answers"]["select"]["choice"], expected,
                        "{backend:?}: {value}"
                    );
                    assert_eq!(value["usage"]["output_tokens"], 0);
                }
            }
            task.abort();
        }
    }

    #[tokio::test]
    async fn package_indexes_fail_closed_before_downloading_model_payloads() {
        for (fault, message) in [
            ("version", "unsupported package schema major"),
            ("directory", "escapes"),
            ("component", "does not match"),
            ("component_path", "escapes"),
            ("shared_path", "escapes"),
            ("json", "parse package index"),
            ("short", "ended before"),
            ("long", "exceeds its declared"),
            ("oversize", "JSON limit"),
            ("hash", "SHA-256"),
            ("lfs", "inconsistent LFS"),
            ("unsafe_sibling", "unsafe"),
        ] {
            let (client, trace, task) =
                variant_hub("tiny-native-engine", false, false, Some(fault)).await;
            let error = client
                .plan(
                    HubModel::new("owner/package:small".into(), "main".into()).unwrap(),
                    HubLimits {
                        bytes: 64 * 1024,
                        files: 16,
                    },
                )
                .await
                .unwrap_err();
            assert!(format!("{error:#}").contains(message), "{fault}: {error:#}");
            let seen = trace.lock().unwrap();
            assert!(
                seen.len() <= 2
                    && seen
                        .iter()
                        .all(|path| !path.contains("/small/") && !path.contains("/large/")),
                "{fault}: {seen:?}"
            );
            task.abort();
        }
    }

    #[tokio::test]
    async fn unpublished_package_variant_is_rejected_before_payload_download() {
        let (client, trace, task) = variant_hub("tiny-native-engine", false, false, None).await;
        let error = client
            .plan(
                HubModel::new("owner/package:unpublished".into(), "main".into()).unwrap(),
                HubLimits {
                    bytes: 64 * 1024,
                    files: 16,
                },
            )
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("variant 'unpublished'"),
            "{error:#}"
        );
        assert_eq!(trace.lock().unwrap().len(), 2);
        task.abort();
    }
}
