//! Immutable Hub download plans. Remote names never become unvalidated local paths.
mod package;
use anyhow::{Context, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, time::Duration};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

const INFO_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const STORED_PLAN_BYTES: u64 = INFO_BYTES as u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DownloadStatus {
    Complete,
    Paused,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct HubProgress<'a> {
    pub(crate) url: &'a str,
    pub(crate) done: u64,
    pub(crate) total: u64,
    pub(crate) package_bytes: u64,
    pub(crate) package_total: u64,
}
async fn until_paused<F: std::future::Future>(
    pause: &mut tokio::sync::watch::Receiver<bool>,
    operation: F,
) -> Option<F::Output> {
    if *pause.borrow() {
        return None;
    }
    tokio::select! {biased; _=pause.wait_for(|value|*value)=>None, result=operation=>Some(result)}
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubModel {
    id: String,
    repository: String,
    variant: Option<String>,
    revision: String,
}
impl HubModel {
    pub(crate) fn new(id: String, revision: String) -> anyhow::Result<Self> {
        let (repository, variant) = match id.split_once(':') {
            Some((repository, variant)) => (repository.to_owned(), Some(variant.to_owned())),
            None => (id, None),
        };
        Self::from_parts(repository, variant, revision)
    }
    fn from_parts(
        repository: String,
        variant: Option<String>,
        revision: String,
    ) -> anyhow::Result<Self> {
        if let Some(variant) = &variant {
            ensure!(
                valid_component(variant),
                "model variant {variant:?} must be a nonempty package variant identifier using letters, digits, '.', '_' or '-'; use an explicitly published manifest variant"
            );
        }
        let parts = repository.split('/').collect::<Vec<_>>();
        ensure!(
            (1..=2).contains(&parts.len()) && parts.iter().all(|part| valid_component(part)),
            "model {repository:?} must be a Hugging Face repository or owner/repository identifier; URLs and local paths are not package identifiers; use owner/repository[:variant] with an explicitly published model-package variant"
        );
        ensure!(
            !revision.is_empty()
                && revision.len() <= 255
                && !revision.chars().any(char::is_control),
            "model revision must be a nonempty Git branch, tag or commit of at most 255 bytes"
        );
        let id = variant.as_ref().map_or_else(
            || repository.clone(),
            |variant| format!("{repository}:{variant}"),
        );
        Ok(Self {
            id,
            repository,
            variant,
            revision,
        })
    }
    pub(crate) fn id(&self) -> &str {
        &self.id
    }
    pub(crate) fn variant(&self) -> Option<&str> {
        self.variant.as_deref()
    }
    pub(crate) fn package_selection(&self) -> onnx_genai_ort::ModelPackageSelection {
        onnx_genai_ort::ModelPackageSelection {
            variant: self.variant.clone(),
            ..Default::default()
        }
    }
}
fn valid_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value != "."
        && value != ".."
        && !value.starts_with('.')
        && !value.ends_with('.')
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HubLimits {
    pub(crate) bytes: u64,
    pub(crate) files: usize,
}
#[derive(Debug)]
struct DownloadFile {
    name: String,
    bytes: u64,
    sha256: Option<String>,
}
#[derive(Debug)]
pub(crate) struct DownloadPlan {
    model: HubModel,
    commit: String,
    files: Vec<DownloadFile>,
    bytes: u64,
}
impl DownloadPlan {
    pub(crate) fn model(&self) -> &HubModel {
        &self.model
    }
    pub(crate) fn id(&self) -> &str {
        self.model.id()
    }
    #[cfg(test)]
    pub(crate) fn commit(&self) -> &str {
        &self.commit
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) fn file_count(&self) -> usize {
        self.files.len()
    }
    pub(crate) fn encode(&self) -> anyhow::Result<Vec<u8>> {
        let siblings = self.files.iter().map(|file| serde_json::json!({
            "rfilename":file.name, "size":file.bytes,
            "lfs":file.sha256.as_ref().map(|hash|serde_json::json!({"size":file.bytes,"sha256":hash}))
        })).collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&serde_json::json!({
            "version":2, "repository":self.model.repository, "variant":self.model.variant, "revision":self.model.revision,
            "sha":self.commit, "siblings":siblings
        }))
        .context("encode immutable cached model manifest")?;
        ensure!(
            bytes.len() <= INFO_BYTES,
            "cached model manifest exceeds {INFO_BYTES} bytes; select a package with fewer or shorter file paths"
        );
        Ok(bytes)
    }
    pub(crate) fn decode(bytes: &[u8], limits: HubLimits) -> anyhow::Result<Self> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Stored {
            version: u32,
            repository: String,
            variant: Option<String>,
            revision: String,
            sha: String,
            siblings: Vec<RepositoryFile>,
        }
        ensure!(
            bytes.len() <= INFO_BYTES,
            "cached model manifest exceeds {INFO_BYTES} bytes; restore a valid bounded cache manifest"
        );
        let stored: Stored = serde_json::from_slice(bytes)
            .context("parse cached model manifest; restore valid version-2 cache metadata")?;
        ensure!(
            stored.version == 2,
            "unsupported cached model manifest version {}; use a compatible nxrt build or a separate cache",
            stored.version
        );
        Self::from_info(
            HubModel::from_parts(stored.repository, stored.variant, stored.revision)?,
            RepositoryInfo {
                sha: stored.sha,
                siblings: stored.siblings,
            },
            limits,
        )
    }
    pub(crate) fn files(&self) -> impl Iterator<Item = (&str, u64)> {
        self.files
            .iter()
            .map(|file| (file.name.as_str(), file.bytes))
    }
    fn from_info(model: HubModel, info: RepositoryInfo, limits: HubLimits) -> anyhow::Result<Self> {
        validate_commit(&info.sha, model.id())?;
        ensure!(
            !info.siblings.is_empty() && info.siblings.len() <= limits.files,
            "Hub model '{}' contains {} files; require 1..={} files under the configured download limit",
            model.id(),
            info.siblings.len(),
            limits.files
        );
        let mut names = BTreeSet::new();
        let mut files = Vec::new();
        let mut bytes = 0_u64;
        for file in info.siblings {
            let name = file.rfilename;
            ensure!(
                valid_file(&name) && names.insert(name.clone()),
                "Hub file {name:?} has an unsafe, reserved or duplicate relative path; publish regular files beneath the package root"
            );
            let size = file.size.context(format!("Hub file {name:?} omits its size; request file metadata with blobs=true before downloading"))?;
            bytes = bytes
                .checked_add(size)
                .context("Hub download byte count overflow; select a smaller package")?;
            ensure!(
                bytes <= limits.bytes,
                "Hub package exceeds configured download limit {} bytes; choose a smaller package or raise --model-cache-max-bytes",
                limits.bytes
            );
            let sha256 = admit_lfs(&name, size, file.lfs.as_ref())?;
            files.push(DownloadFile {
                name,
                bytes: size,
                sha256,
            });
        }
        ensure!(
            onnx_genai_metadata::METADATA_FILE_NAMES
                .iter()
                .any(|name| names.contains(*name))
                || names.contains("genai_config.json")
                || names.contains("manifest.json"),
            "Hub model '{}' is not an nxrt model package: no supported inference_metadata.yaml/yml/json, genai_config.json or model-package manifest.json; export an ONNX package with explicit metadata before downloading it for serving",
            model.id()
        );
        ensure!(
            model.variant().is_none() || names.contains("manifest.json"),
            "Hub model '{}' requests a named variant but has no model-package manifest.json; publish the selected variant explicitly instead of inferring a graph or quantization from filenames",
            model.id()
        );
        Ok(Self {
            model,
            commit: info.sha,
            files,
            bytes,
        })
    }
}
fn admit_lfs(name: &str, size: u64, lfs: Option<&LfsFile>) -> anyhow::Result<Option<String>> {
    lfs.map(|lfs| {
        ensure!(lfs.size == size && lfs.sha256.len() == 64 && lfs.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Hub file {name:?} has inconsistent LFS size/hash metadata; repair the repository before downloading");
        Ok(lfs.sha256.to_ascii_lowercase())
    }).transpose()
}

fn validate_commit(commit: &str, model: &str) -> anyhow::Result<()> {
    ensure!(
        commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Hub model {model:?} returned an invalid immutable Git commit; retry or repair the Hub endpoint"
    );
    Ok(())
}

fn valid_file(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 4096
        && !name.contains(['\\', ':'])
        && !name.chars().any(char::is_control)
        && name.split('/').all(|part| {
            !part.is_empty() && part != "." && part != ".." && !part.starts_with(".nxrt-")
        })
}

#[derive(Deserialize)]
struct RepositoryInfo {
    sha: String,
    siblings: Vec<RepositoryFile>,
}
#[derive(Deserialize)]
struct RepositoryFile {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<LfsFile>,
}
#[derive(Deserialize)]
struct LfsFile {
    size: u64,
    sha256: String,
}

pub(crate) struct HubClient {
    client: reqwest::Client,
    endpoint: reqwest::Url,
}
impl HubClient {
    pub(crate) fn new() -> anyhow::Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(token) = std::env::var("HF_TOKEN") {
            let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .context("HF_TOKEN cannot be represented as an HTTP header; supply a valid Hub access token")?;
            value.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        Ok(Self {
            client: reqwest::Client::builder()
                .default_headers(headers)
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(60))
                .build()
                .context("initialize Hugging Face download client")?,
            endpoint: reqwest::Url::parse("https://huggingface.co/").expect("static Hub URL"),
        })
    }
    fn url(&self, segments: &[&str]) -> reqwest::Url {
        let mut url = self.endpoint.clone();
        url.path_segments_mut()
            .expect("Hub base URL supports path segments")
            .pop_if_empty()
            .extend(segments);
        url
    }
    pub(crate) async fn plan(
        &self,
        model: HubModel,
        limits: HubLimits,
    ) -> anyhow::Result<DownloadPlan> {
        let mut segments = vec!["api", "models"];
        segments.extend(model.repository.split('/'));
        segments.extend(["revision", &model.revision]);
        let mut url = self.url(&segments);
        url.query_pairs_mut().append_pair("blobs", "true");
        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| {
                format!(
                    "fetch download metadata for model '{}'; check network and HF_TOKEN",
                    model.id()
                )
            })?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Hub refused model '{}' revision {:?}; check repository access and revision",
                    model.id(),
                    model.revision
                )
            })?;
        let mut data = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("read Hub repository metadata")?
        {
            ensure!(
                chunk.len() <= INFO_BYTES.saturating_sub(data.len()),
                "Hub repository metadata exceeds {INFO_BYTES} bytes; select a smaller package"
            );
            data.extend_from_slice(&chunk);
        }
        let mut info: RepositoryInfo = serde_json::from_slice(&data)
            .context("parse Hub repository metadata; expected sha and sized file siblings")?;
        self.select_package_files(&model, &mut info).await?;
        DownloadPlan::from_info(model, info, limits)
    }
    pub(crate) fn file_url(&self, plan: &DownloadPlan, name: &str) -> reqwest::Url {
        self.immutable_file_url(&plan.model, &plan.commit, name)
    }
    fn immutable_file_url(&self, model: &HubModel, commit: &str, name: &str) -> reqwest::Url {
        let mut segments = model.repository.split('/').collect::<Vec<_>>();
        segments.extend(["resolve", commit]);
        segments.extend(name.split('/'));
        self.url(&segments)
    }

    /// Resume only an admitted, exclusively borrowed cache transfer.
    pub(crate) async fn download(
        &self,
        transfer: &mut crate::model_cache::PendingTransfer,
        pause: &mut tokio::sync::watch::Receiver<bool>,
        progress: impl Fn(HubProgress<'_>),
    ) -> anyhow::Result<DownloadStatus> {
        let plan = std::sync::Arc::clone(transfer.plan());
        let mut completed = 0_u64;
        for file in &plan.files {
            if *pause.borrow() {
                return Ok(DownloadStatus::Paused);
            }
            let url = self.file_url(&plan, &file.name);
            let mut output = transfer.open_file(&file.name).await?;
            let mut downloaded = output.metadata().await?.len();
            ensure!(
                downloaded <= file.bytes,
                "owned staged file {:?} exceeds its declared {} bytes; cancel this download and recreate it",
                file.name,
                file.bytes
            );
            let base = completed;
            let report = |done| {
                progress(HubProgress {
                    url: url.as_str(),
                    done,
                    total: file.bytes,
                    package_bytes: base + done,
                    package_total: plan.bytes,
                })
            };
            report(downloaded);
            let mut hash = Sha256::new();
            let mut remaining = downloaded;
            let mut buffer = [0_u8; 64 * 1024];
            output.seek(std::io::SeekFrom::Start(0)).await?;
            while remaining > 0 {
                if *pause.borrow() {
                    return Ok(DownloadStatus::Paused);
                }
                let count = usize::try_from(remaining.min(buffer.len() as u64))
                    .expect("bounded prefix buffer");
                let read = output
                    .read(&mut buffer[..count])
                    .await
                    .with_context(|| format!("read staged prefix for {:?}", file.name))?;
                ensure!(
                    read > 0,
                    "staged prefix for {:?} changed length while resuming; cancel and recreate the download",
                    file.name
                );
                hash.update(&buffer[..read]);
                remaining -= read as u64;
            }
            if downloaded == file.bytes {
                if file
                    .sha256
                    .as_ref()
                    .is_none_or(|expected| format!("{:x}", hash.clone().finalize()) == *expected)
                {
                    completed += downloaded;
                    continue;
                }
                // This is our own immutable-plan transfer, so a corrupt complete file can be retried safely.
                output.set_len(0).await?;
                output.seek(std::io::SeekFrom::Start(0)).await?;
                downloaded = 0;
                hash = Sha256::new();
                report(0);
            }
            let mut request = self.client.get(url.clone());
            if downloaded > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={downloaded}-"));
            }
            let Some(response) = until_paused(pause, request.send()).await else {
                output.sync_all().await?;
                return Ok(DownloadStatus::Paused);
            };
            let mut response = response
                .with_context(|| {
                    format!(
                        "download pinned Hub file {:?} for model '{}'; check network and HF_TOKEN",
                        file.name,
                        plan.id()
                    )
                })?
                .error_for_status()
                .with_context(|| {
                    format!(
                        "Hub refused pinned file {:?} at {}; retry the download",
                        file.name, plan.commit
                    )
                })?;
            match response.status() {
                reqwest::StatusCode::OK => {
                    if downloaded > 0 {
                        output.set_len(0).await?;
                        output.seek(std::io::SeekFrom::Start(0)).await?;
                        downloaded = 0;
                        hash = Sha256::new();
                        report(0);
                    }
                }
                reqwest::StatusCode::PARTIAL_CONTENT => {
                    let actual = response
                        .headers()
                        .get(reqwest::header::CONTENT_RANGE)
                        .and_then(|value| value.to_str().ok());
                    ensure!(
                        file.bytes > 0,
                        "Hub returned a partial response for empty file {:?}; require a complete empty response",
                        file.name
                    );
                    let expected =
                        format!("bytes {}-{}/{}", downloaded, file.bytes - 1, file.bytes);
                    ensure!(
                        actual == Some(expected.as_str()),
                        "Hub file {:?} returned Content-Range {:?}, expected {:?}; retry without changing the immutable plan",
                        file.name,
                        actual,
                        expected
                    );
                }
                status => anyhow::bail!(
                    "Hub file {:?} returned unexpected status {status}; require a complete 200 response or validated 206 range",
                    file.name
                ),
            }
            if let Some(length) = response.content_length() {
                ensure!(
                    length <= file.bytes - downloaded,
                    "Hub file {:?} exceeds its declared {} bytes; reject the response before writing",
                    file.name,
                    file.bytes
                );
            }
            loop {
                let Some(chunk) = until_paused(pause, response.chunk()).await else {
                    output.sync_all().await?;
                    return Ok(DownloadStatus::Paused);
                };
                let Some(chunk) = chunk.with_context(|| {
                    format!(
                        "read Hub file {:?}; resume this incomplete download",
                        file.name
                    )
                })?
                else {
                    break;
                };
                ensure!(
                    chunk.len() as u64 <= file.bytes.saturating_sub(downloaded),
                    "Hub file {:?} exceeds its declared {} bytes; discard the response and retry the immutable plan",
                    file.name,
                    file.bytes
                );
                // Cancellation is cooperative: never interrupt an in-flight filesystem write.
                output.write_all(&chunk).await.with_context(|| {
                    format!(
                        "write staged Hub file {:?}; check cache disk capacity",
                        file.name
                    )
                })?;
                hash.update(&chunk);
                downloaded += chunk.len() as u64;
                report(downloaded);
            }
            output
                .sync_all()
                .await
                .with_context(|| format!("persist staged Hub file {:?}", file.name))?;
            ensure!(
                downloaded == file.bytes,
                "Hub file {:?} ended at {downloaded} bytes, expected {}; resume the incomplete download",
                file.name,
                file.bytes
            );
            if let Some(expected) = &file.sha256
                && format!("{:x}", hash.finalize()) != *expected
            {
                output.set_len(0).await?;
                output.sync_all().await?;
                report(0);
                anyhow::bail!(
                    "Hub file {:?} does not match its LFS SHA-256; its corrupt owned prefix was reset, retry this download",
                    file.name
                );
            }
            completed += downloaded;
        }
        Ok(DownloadStatus::Complete)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(feature = "native-backend")]
    pub(crate) use super::package::tests::variant_hub;
    use super::*;
    use serde_json::json;
    use std::path::Path;

    fn model() -> HubModel {
        HubModel::new("owner/package".into(), "main".into()).unwrap()
    }
    fn limits() -> HubLimits {
        HubLimits {
            bytes: 1024 * 1024,
            files: 16,
        }
    }
    fn info() -> serde_json::Value {
        json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"inference_metadata.yaml","size":8}]})
    }
    fn parse(value: serde_json::Value, limits: HubLimits) -> anyhow::Result<DownloadPlan> {
        DownloadPlan::from_info(model(), serde_json::from_value(value).unwrap(), limits)
    }
    #[test]
    fn persisted_package_selectors_have_one_identity_and_an_explicit_cache_version() {
        let plan = parse(info(), limits()).unwrap();
        let encoded = plan.encode().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(value["version"], 2);
        for version in [1, 3] {
            value["version"] = json!(version);
            let error =
                DownloadPlan::decode(&serde_json::to_vec(&value).unwrap(), limits()).unwrap_err();
            assert!(
                format!("{error:#}").contains("unsupported cached model manifest version"),
                "{error:#}"
            );
        }
        value["version"] = json!(2);
        value["repository"] = json!("owner/package:hidden");
        let error =
            DownloadPlan::decode(&serde_json::to_vec(&value).unwrap(), limits()).unwrap_err();
        assert!(format!("{error:#}").contains("repository"), "{error:#}");
        value["repository"] = json!("owner/package");
        value["variant"] = json!("");
        let error =
            DownloadPlan::decode(&serde_json::to_vec(&value).unwrap(), limits()).unwrap_err();
        assert!(format!("{error:#}").contains("model variant"), "{error:#}");
    }

    #[test]
    fn hub_identifiers_and_remote_paths_cannot_escape_the_package() {
        for name in [
            "../package",
            "owner/../package",
            "https://host/repo",
            "owner/repo:../Q4",
            "owner/repo:",
            "owner/repo:Q4:extra",
            "owner/repo\\file",
            "owner/",
            "owner/..",
        ] {
            assert!(HubModel::new(name.into(), "main".into()).is_err(), "{name}");
        }
        for name in [
            "../weights",
            "/weights",
            "a/../../weights",
            "a\\weights",
            "a//weights",
            "a/./weights",
            ".nxrt-manifest.json",
            "a/.nxrt-secret",
            "a\nfile",
            "C:/weights",
        ] {
            assert!(!valid_file(name), "{name}");
        }
        assert!(HubModel::new("package".into(), "main".into()).is_ok());
        assert!(valid_file("graphs/decoder.onnx.data"));
        assert!(valid_file(".gitattributes"));
        let client = HubClient {
            client: reqwest::Client::new(),
            endpoint: reqwest::Url::parse("https://huggingface.co/").unwrap(),
        };
        assert_eq!(
            client
                .url(&["api", "models", "owner", "package", "revision", "refs/pr/1"])
                .as_str(),
            "https://huggingface.co/api/models/owner/package/revision/refs%2Fpr%2F1"
        );
    }
    #[test]
    fn hub_plans_require_sized_bounded_immutable_packages() {
        let plan = parse(info(), limits()).unwrap();
        assert_eq!(plan.bytes(), 8);
        assert_eq!(plan.id(), "owner/package");
        let mut yaml_short = info();
        yaml_short["siblings"][0]["rfilename"] = json!("inference_metadata.yml");
        assert!(parse(yaml_short, limits()).is_ok());
        for (value, bounds, needle) in [
            (
                json!({"sha":"main","siblings":[]}),
                limits(),
                "immutable Git commit",
            ),
            (
                json!({"sha":"a".repeat(40),"siblings":[]}),
                limits(),
                "files",
            ),
            (
                json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"model.gguf","size":8}]}),
                limits(),
                "not an nxrt model package",
            ),
            (
                json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"inference_metadata.yaml"}]}),
                limits(),
                "omits its size",
            ),
            (
                info(),
                HubLimits {
                    bytes: 7,
                    files: 16,
                },
                "download limit",
            ),
            (
                json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"inference_metadata.yaml","size":8},{"rfilename":"inference_metadata.yaml","size":8}]}),
                limits(),
                "duplicate",
            ),
            (
                json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"inference_metadata.yaml","size":8,"lfs":{"size":9,"sha256":"b".repeat(64)}}]}),
                limits(),
                "inconsistent LFS",
            ),
        ] {
            let error = parse(value, bounds).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) enum Transfer {
        Valid,
        Corrupt,
        Short,
        Long,
        Range,
        WrongRange,
        FailOnce,
        Slow,
    }
    pub(crate) async fn hub(transfer: Transfer) -> (HubClient, tokio::task::JoinHandle<()>) {
        use axum::{Router, extract::Request, response::IntoResponse, routing::get};
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/decision");
        let mut files = std::collections::BTreeMap::new();
        let mut siblings = Vec::new();
        for name in [
            "inference_metadata.yaml",
            "tokenizer.json",
            "model.onnx.textproto",
        ] {
            let mut bytes = std::fs::read(fixture.join(name)).unwrap();
            siblings.push(json!({"rfilename":name,"size":bytes.len(),"lfs":{"size":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}}));
            if name == "tokenizer.json" {
                match transfer {
                    Transfer::Valid
                    | Transfer::Range
                    | Transfer::WrongRange
                    | Transfer::FailOnce
                    | Transfer::Slow => {}
                    Transfer::Corrupt => bytes[0] ^= 1,
                    Transfer::Short => {
                        bytes.pop();
                    }
                    Transfer::Long => bytes.push(0),
                }
            }
            files.insert(name.to_owned(), bytes);
        }
        let info = json!({"sha":"a".repeat(40),"siblings":siblings});
        let metadata_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed_once = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let router = Router::new()
            .route(
                "/api/models/owner/package/revision/main",
                get(move || {
                    let mut info = info.clone();
                    if matches!(transfer, Transfer::FailOnce)
                        && metadata_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0
                    {
                        info["sha"] = serde_json::json!("b".repeat(40));
                    }
                    async move { axum::Json(info) }
                }),
            )
            .fallback_service(get(move |request: Request| {
                let files = files.clone();
                let failed_once = std::sync::Arc::clone(&failed_once);
                async move {
                    let prefix = format!("/owner/package/resolve/{}/", "a".repeat(40));
                    match request
                        .uri()
                        .path()
                        .strip_prefix(&prefix)
                        .and_then(|name| files.get(name))
                    {
                        Some(bytes) => {
                            let tokenizer = request.uri().path().ends_with("/tokenizer.json");
                            if tokenizer
                                && matches!(transfer, Transfer::FailOnce)
                                && !failed_once.swap(true, std::sync::atomic::Ordering::SeqCst)
                            {
                                return bytes[..bytes.len() - 1].to_vec().into_response();
                            }
                            let range_present =
                                request.headers().contains_key(axum::http::header::RANGE);
                            if tokenizer && matches!(transfer, Transfer::Slow) && !range_present {
                                let (tx, rx) = tokio::sync::mpsc::channel::<
                                    Result<axum::body::Bytes, std::convert::Infallible>,
                                >(2);
                                let first = bytes[..37].to_vec();
                                tokio::spawn(async move {
                                    let _ = tx.send(Ok(first.into())).await;
                                    tx.closed().await;
                                });
                                return axum::body::Body::from_stream(
                                    tokio_stream::wrappers::ReceiverStream::new(rx),
                                )
                                .into_response();
                            }
                            let is_range = matches!(
                                transfer,
                                Transfer::Range
                                    | Transfer::WrongRange
                                    | Transfer::FailOnce
                                    | Transfer::Slow
                            );
                            let range = request
                                .headers()
                                .get(axum::http::header::RANGE)
                                .and_then(|value| value.to_str().ok());
                            if matches!(transfer, Transfer::Range | Transfer::WrongRange)
                                && request.uri().path().ends_with("/tokenizer.json")
                                && range.is_none()
                            {
                                return axum::http::StatusCode::PRECONDITION_FAILED.into_response();
                            }
                            if is_range
                                && let Some(start) = range
                                    .and_then(|range| range.strip_prefix("bytes="))
                                    .and_then(|range| range.strip_suffix('-'))
                                    .and_then(|start| start.parse::<usize>().ok())
                            {
                                if start >= bytes.len() {
                                    return axum::http::StatusCode::RANGE_NOT_SATISFIABLE
                                        .into_response();
                                }
                                let header_start = if matches!(transfer, Transfer::WrongRange) {
                                    0
                                } else {
                                    start
                                };
                                return (
                                    axum::http::StatusCode::PARTIAL_CONTENT,
                                    [(
                                        axum::http::header::CONTENT_RANGE,
                                        format!(
                                            "bytes {}-{}/{}",
                                            header_start,
                                            bytes.len() - 1,
                                            bytes.len()
                                        ),
                                    )],
                                    bytes[start..].to_vec(),
                                )
                                    .into_response();
                            }
                            bytes.clone().into_response()
                        }
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
            task,
        )
    }
    fn owned(
        plan: DownloadPlan,
    ) -> (
        tempfile::TempDir,
        crate::model_cache::ModelCache,
        crate::model_cache::PendingTransfer,
    ) {
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
        (root, cache, transfer)
    }
    #[tokio::test]
    async fn immutable_download_preserves_a_loadable_native_package_and_reports_bytes() {
        let (client, task) = hub(Transfer::Valid).await;
        let plan = client.plan(model(), limits()).await.unwrap();
        let total = plan.bytes();
        let (_root, _cache, mut transfer) = owned(plan);
        let (_control, mut pause) = tokio::sync::watch::channel(false);
        let progress = std::sync::atomic::AtomicU64::new(0);
        assert_eq!(
            client
                .download(&mut transfer, &mut pause, |state| {
                    progress.store(state.package_bytes, std::sync::atomic::Ordering::Relaxed);
                    assert!(state.done <= state.total);
                    assert_eq!(state.package_total, total);
                    assert!(state.url.contains(&"a".repeat(40)));
                })
                .await
                .unwrap(),
            DownloadStatus::Complete
        );
        assert_eq!(progress.load(std::sync::atomic::Ordering::Relaxed), total);
        crate::AppState::load(transfer.path(), Some("downloaded-fixture".into())).unwrap();
        transfer.cancel().unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn corrupt_downloads_fail_hash_verification_before_publication() {
        let (client, task) = hub(Transfer::Corrupt).await;
        let plan = client.plan(model(), limits()).await.unwrap();
        let (_root, _cache, mut transfer) = owned(plan);
        let (_control, mut pause) = tokio::sync::watch::channel(false);
        let error = client
            .download(&mut transfer, &mut pause, |_| {})
            .await
            .unwrap_err();
        assert!(error.to_string().contains("LFS SHA-256"), "{error:#}");
        transfer.cancel().unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn transfer_lengths_and_owned_prefixes_are_checked_before_publication() {
        for (mode, needle) in [
            (Transfer::Short, "ended at"),
            (Transfer::Long, "exceeds its declared"),
        ] {
            let (client, task) = hub(mode).await;
            let plan = client.plan(model(), limits()).await.unwrap();
            let (_root, _cache, mut transfer) = owned(plan);
            let (_control, mut pause) = tokio::sync::watch::channel(false);
            let error = client
                .download(&mut transfer, &mut pause, |_| {})
                .await
                .unwrap_err();
            assert!(error.to_string().contains(needle), "{error:#}");
            transfer.cancel().unwrap();
            task.abort();
        }
        let (client, task) = hub(Transfer::Valid).await;
        let plan = client.plan(model(), limits()).await.unwrap();
        let (_root, _cache, mut transfer) = owned(plan);
        std::fs::write(transfer.path().join("inference_metadata.yaml"), "existing").unwrap();
        let (_control, mut pause) = tokio::sync::watch::channel(false);
        assert_eq!(
            client
                .download(&mut transfer, &mut pause, |_| {})
                .await
                .unwrap(),
            DownloadStatus::Complete
        );
        assert_ne!(
            std::fs::read_to_string(transfer.path().join("inference_metadata.yaml")).unwrap(),
            "existing"
        );
        transfer.cancel().unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn resumes_require_correct_ranges_and_preserve_prefixes_on_invalid_responses() {
        for mode in [Transfer::Range, Transfer::WrongRange] {
            let (client, task) = hub(mode).await;
            let plan = client.plan(model(), limits()).await.unwrap();
            let (_root, _cache, mut transfer) = owned(plan);
            let tokenizer = std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/onnx_genai_workflows/decision/tokenizer.json"),
            )
            .unwrap();
            let path = transfer.path().join("tokenizer.json");
            std::fs::write(&path, &tokenizer[..37]).unwrap();
            let (_control, mut pause) = tokio::sync::watch::channel(false);
            let result = client.download(&mut transfer, &mut pause, |_| {}).await;
            match mode {
                Transfer::Range => {
                    assert_eq!(result.unwrap(), DownloadStatus::Complete);
                    assert_eq!(std::fs::read(&path).unwrap(), tokenizer);
                }
                Transfer::WrongRange => {
                    assert!(result.unwrap_err().to_string().contains("Content-Range"));
                    assert_eq!(std::fs::read(&path).unwrap(), tokenizer[..37]);
                }
                _ => unreachable!(),
            }
            transfer.cancel().unwrap();
            task.abort();
        }
    }
}
