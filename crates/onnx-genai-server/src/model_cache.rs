//! Server-owned, durable package transfers and atomic cache publications.
use crate::{
    model_hub::{
        DownloadPlan, DownloadStatus, HubClient, HubLimits, HubProgress, STORED_PLAN_BYTES,
    },
    models_config::ModelSpec,
};
use anyhow::{Context, ensure};
use cap_std::fs::{Dir, OpenOptions};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
const LOCK: &str = ".nxrt-models.lock";
const MANIFEST: &str = ".nxrt-model.json";
#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheLimits {
    pub(crate) bytes: u64,
    pub(crate) files: usize,
    pub(crate) models: usize,
}
#[derive(Debug, Clone, Copy, Default)]
struct Usage {
    bytes: u64,
    files: usize,
    models: usize,
}
#[derive(Debug, thiserror::Error)]
#[error(
    "model cache quota exhausted: {detail}; remove cached models or raise the model-cache limits"
)]
pub(crate) struct CacheQuota {
    detail: String,
}
impl Usage {
    fn add(self, other: Self) -> anyhow::Result<Self> {
        Ok(Self {
            bytes: self
                .bytes
                .checked_add(other.bytes)
                .context("model cache byte accounting overflow")?,
            files: self
                .files
                .checked_add(other.files)
                .context("model cache file accounting overflow")?,
            models: self
                .models
                .checked_add(other.models)
                .context("model cache model accounting overflow")?,
        })
    }
    fn subtract(&mut self, other: Self) {
        self.bytes = self
            .bytes
            .checked_sub(other.bytes)
            .expect("owned cache byte credit");
        self.files = self
            .files
            .checked_sub(other.files)
            .expect("owned cache file credit");
        self.models = self
            .models
            .checked_sub(other.models)
            .expect("owned cache model credit");
    }
    fn admit(self, limits: CacheLimits) -> anyhow::Result<()> {
        if self.bytes > limits.bytes || self.files > limits.files || self.models > limits.models {
            return Err(CacheQuota{detail:format!("need {} bytes, {} files and {} models; limits are {} bytes, {} files and {} models",self.bytes,self.files,self.models,limits.bytes,limits.files,limits.models)}.into());
        }
        Ok(())
    }
}
struct CacheRoot {
    directory: Dir,
    path: PathBuf,
    _process_lock: std::fs::File,
    limits: CacheLimits,
    usage: Mutex<Usage>,
}
#[derive(Clone)]
pub(crate) struct ModelCache(Arc<CacheRoot>);
pub(crate) struct PublishedPackage {
    root: Arc<CacheRoot>,
    directory: String,
    plan: Arc<DownloadPlan>,
    usage: Usage,
}
impl PublishedPackage {
    pub(crate) fn request(&self) -> &crate::model_hub::HubModel {
        self.plan.model()
    }
    pub(crate) fn id(&self) -> &str {
        self.plan.id()
    }
    #[cfg(test)]
    pub(crate) fn commit(&self) -> &str {
        self.plan.commit()
    }
    pub(crate) fn spec(&self) -> ModelSpec {
        ModelSpec {
            id: self.id().into(),
            path: self.root.path.join(&self.directory),
            eager: false,
            warmup: false,
        }
    }
}
pub(crate) struct CacheOperationError {
    pub(crate) package: Arc<PublishedPackage>,
    pub(crate) cause: anyhow::Error,
}
impl std::fmt::Debug for CacheOperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheOperationError")
            .field("model", &self.package.id())
            .field("cause", &self.cause)
            .finish()
    }
}
impl std::fmt::Display for CacheOperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for CacheOperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
/// A unique, persisted transfer. Dropping its owner preserves the files and quota.
pub(crate) struct PendingTransfer {
    stage: Stage,
}
struct Stage {
    root: Arc<CacheRoot>,
    directory: String,
    path: PathBuf,
    plan: Arc<DownloadPlan>,
    credit: Usage,
}
pub(crate) struct CompletedTransfer {
    stage: Stage,
}
pub(crate) struct TransferFailure {
    pub(crate) transfer: PendingTransfer,
    pub(crate) cause: anyhow::Error,
}
impl std::fmt::Debug for TransferFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferFailure")
            .field("model", &self.transfer.id())
            .field("cause", &self.cause)
            .finish()
    }
}
pub(crate) enum TransferOutcome {
    Complete(CompletedTransfer),
    Paused(PendingTransfer),
    Failed(TransferFailure),
}
impl PendingTransfer {
    pub(crate) fn id(&self) -> &str {
        self.stage.plan.id()
    }
    pub(crate) fn plan(&self) -> &Arc<DownloadPlan> {
        &self.stage.plan
    }
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.stage.path
    }
    pub(crate) async fn run(
        mut self,
        client: &HubClient,
        pause: &mut tokio::sync::watch::Receiver<bool>,
        progress: impl Fn(HubProgress<'_>),
    ) -> TransferOutcome {
        match client.download(&mut self, pause, progress).await {
            Ok(DownloadStatus::Complete) => {
                TransferOutcome::Complete(CompletedTransfer { stage: self.stage })
            }
            Ok(DownloadStatus::Paused) => TransferOutcome::Paused(self),
            Err(cause) => TransferOutcome::Failed(TransferFailure {
                transfer: self,
                cause,
            }),
        }
    }
    pub(crate) async fn open_file(&mut self, name: &str) -> anyhow::Result<tokio::fs::File> {
        let root = Arc::clone(&self.stage.root);
        let directory = self.stage.directory.clone();
        let name = name.to_owned();
        tokio::task::spawn_blocking(move || {
            let dir = root
                .directory
                .open_dir(&directory)
                .context("open owned transfer directory")?;
            if let Some(parent) = Path::new(&name).parent() {
                dir.create_dir_all(parent)
                    .with_context(|| format!("create parent for staged file {name:?}"))?;
            }
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true);
            #[cfg(unix)]
            {
                use cap_std::fs::OpenOptionsExt;
                options
                    .mode(0o600)
                    .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
            }
            let file = dir.open_with(&name, &options).with_context(|| {
                format!("open owned staged file {name:?}; repair cache permissions")
            })?;
            ensure!(
                file.metadata()?.is_file(),
                "staged file {name:?} is not regular; cancel this download and recreate it"
            );
            let file = file.into_std();
            file.try_lock().with_context(|| format!(
                "staged file {name:?} is still owned by another disk operation; wait for it to finish before resuming"
            ))?;
            Ok::<_, anyhow::Error>(file)
        })
        .await
        .context("staged file open task panicked")?
        .map(tokio::fs::File::from_std)
    }
    pub(crate) fn cancel(self) -> Result<(), TransferFailure> {
        match self.stage.root.remove_directory(
            &self.stage.directory,
            &self.stage.plan,
            self.stage.credit,
        ) {
            Ok(()) => Ok(()),
            Err(cause) => Err(TransferFailure {
                transfer: self,
                cause,
            }),
        }
    }
}
impl CompletedTransfer {
    /// Blocking publication; callers run this on their owned blocking task.
    pub(crate) fn publish(self) -> Result<Arc<PublishedPackage>, TransferFailure> {
        self.stage.publish()
    }
}
fn record_name(prefix: &str, id: &str) -> String {
    format!("{prefix}{:x}", Sha256::digest(id.as_bytes()))
}
fn package_name(id: &str) -> String {
    record_name("package-", id)
}
fn stage_name(id: &str) -> String {
    record_name(".nxrt-stage-", id)
}
fn usage_for(plan: &DownloadPlan, manifest_bytes: usize) -> anyhow::Result<Usage> {
    Usage {
        bytes: plan.bytes(),
        files: plan.file_count(),
        models: 1,
    }
    .add(Usage {
        bytes: manifest_bytes as u64,
        files: 1,
        models: 0,
    })
}
impl ModelCache {
    pub(crate) fn open(
        path: &Path,
        limits: CacheLimits,
    ) -> anyhow::Result<(Self, Vec<Arc<PublishedPackage>>, Vec<PendingTransfer>)> {
        ensure!(
            limits.bytes > 0 && limits.files > 0 && limits.models > 0,
            "model cache limits must be positive; configure positive byte, file and model limits"
        );
        std::fs::create_dir_all(path).with_context(|| {
            format!(
                "create model cache '{}'; choose a writable operator-owned directory",
                path.display()
            )
        })?;
        let path = path
            .canonicalize()
            .context("resolve model cache directory")?;
        let directory = Dir::open_ambient_dir(&path, cap_std::ambient_authority())
            .context("open model cache capability directory")?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let lock = directory
            .open_with(LOCK, &options)
            .context("open model cache process lock")?
            .into_std();
        ensure!(
            lock.metadata()?.is_file() && lock.metadata()?.len() == 0,
            "invalid model cache lock; use an empty regular .nxrt-models.lock file"
        );
        lock.try_lock().context("lock model cache; another server owns this directory, choose a separate cache or stop that server")?;
        let cache = Self(Arc::new(CacheRoot {
            directory,
            path,
            _process_lock: lock,
            limits,
            usage: Mutex::new(Usage::default()),
        }));
        let names = cache
            .0
            .directory
            .entries()?
            .map(|entry| {
                entry?.file_name().into_string().map_err(|_| {
                    anyhow::anyhow!(
                        "non-UTF-8 model cache entry; use a dedicated managed directory"
                    )
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut packages = Vec::new();
        let mut pending = Vec::new();
        let mut usage = Usage::default();
        let mut ids = std::collections::BTreeSet::new();
        for mut name in names {
            if name == LOCK {
                continue;
            }
            ensure!(
                cache.0.directory.symlink_metadata(&name)?.is_dir(),
                "unrecognized model cache entry {name:?}; preserve operator files and use a dedicated cache"
            );
            let dir = cache.0.directory.open_dir(&name)?;
            if name.starts_with(".nxrt-trash-") && dir.entries()?.next().is_none() {
                cache.0.directory.remove_dir(&name)?;
                continue;
            }
            if name.starts_with(".nxrt-prepare-") {
                let children = dir
                    .entries()?
                    .map(|entry| Ok(entry?.file_name()))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                ensure!(
                    children.iter().all(|child| child == MANIFEST),
                    "unexpected files in incomplete cache initialization {name:?}; preserve them and repair the cache"
                );
                // No artifact transfer starts until the plan directory has its stable name.
                cache
                    .0
                    .directory
                    .remove_dir_all(&name)
                    .context("recover interrupted download initialization")?;
                continue;
            }
            ensure!(
                name.starts_with("package-")
                    || name.starts_with(".nxrt-stage-")
                    || name.starts_with(".nxrt-trash-"),
                "unrecognized model cache entry {name:?}; preserve operator files and use a dedicated cache"
            );
            let meta = dir
                .symlink_metadata(MANIFEST)
                .with_context(|| format!("inspect cache manifest in {name:?}"))?;
            ensure!(
                meta.is_file() && meta.len() <= STORED_PLAN_BYTES,
                "invalid cache manifest in {name:?}; restore a regular bounded manifest"
            );
            let mut bytes = Vec::new();
            dir.open(MANIFEST)?
                .take(STORED_PLAN_BYTES + 1)
                .read_to_end(&mut bytes)?;
            // A smaller new admission quota must not prevent cleanup of an
            // already committed deletion. Manifest size/path checks still apply.
            let manifest_limits = if name.starts_with(".nxrt-trash-") {
                HubLimits {
                    bytes: u64::MAX,
                    files: usize::MAX,
                }
            } else {
                cache.hub_limits()
            };
            let plan = Arc::new(DownloadPlan::decode(&bytes, manifest_limits)?);
            if name.starts_with(".nxrt-trash-") {
                cache.0.purge(&name, &plan).context(
                    "recover interrupted model deletion; repair cache permissions before retrying",
                )?;
                continue;
            }
            let partial = name.starts_with(".nxrt-stage-");
            ensure!(
                name == if partial {
                    stage_name(plan.id())
                } else {
                    package_name(plan.id())
                },
                "cache directory {name:?} does not match its immutable model id; restore the original entry"
            );
            ensure!(
                ids.insert(plan.id().to_owned()),
                "cache has multiple entries for model '{}'; recover the duplicated stage before startup",
                plan.id()
            );
            verify_files(&dir, &plan, partial)?;
            let credit = usage_for(&plan, bytes.len())?;
            usage = usage.add(credit)?;
            usage.admit(limits)?;
            if partial {
                pending.push(PendingTransfer {
                    stage: Stage {
                        root: Arc::clone(&cache.0),
                        path: cache.0.path.join(&name),
                        directory: std::mem::take(&mut name),
                        plan,
                        credit,
                    },
                });
            } else {
                packages.push(Arc::new(PublishedPackage {
                    root: Arc::clone(&cache.0),
                    directory: name,
                    plan,
                    usage: credit,
                }));
            }
        }
        *cache
            .0
            .usage
            .lock()
            .map_err(|_| anyhow::anyhow!("model cache accounting lock poisoned"))? = usage;
        Ok((cache, packages, pending))
    }
    pub(crate) fn resolve_configured_path(&self, path: &Path) -> anyhow::Result<PathBuf> {
        let mut candidate = std::path::absolute(path)
            .with_context(|| format!("resolve configured model path '{}'", path.display()))?;
        let mut remainder = Vec::new();
        let resolved = loop {
            match candidate.canonicalize() {
                Ok(mut resolved) => {
                    for part in remainder.iter().rev() {
                        if part == std::ffi::OsStr::new("..") {
                            resolved.pop();
                        } else {
                            resolved.push(part);
                        }
                    }
                    break resolved;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let part = candidate
                        .components()
                        .next_back()
                        .context("configured model path has no resolvable parent")?;
                    match part {
                        std::path::Component::Normal(name) => remainder.push(name.to_os_string()),
                        std::path::Component::ParentDir => remainder.push("..".into()),
                        std::path::Component::CurDir => {}
                        _ => return Err(error).context("resolve model path filesystem root"),
                    }
                    candidate.pop();
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "resolve configured model path '{}'; check directory permissions",
                            path.display()
                        )
                    });
                }
            }
        };
        ensure!(
            !resolved.starts_with(&self.0.path),
            "configured model path '{}' lies inside the managed cache; use its discovered cache model id instead of configuring a second owner for those files",
            path.display()
        );
        Ok(resolved)
    }
    pub(crate) fn hub_limits(&self) -> HubLimits {
        HubLimits {
            bytes: self.0.limits.bytes,
            files: self.0.limits.files,
        }
    }
    pub(crate) fn begin(&self, plan: DownloadPlan) -> anyhow::Result<PendingTransfer> {
        let manifest = plan.encode()?;
        let credit = usage_for(&plan, manifest.len())?;
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| anyhow::anyhow!("model cache accounting lock poisoned"))?;
        let total = usage.add(credit)?;
        total.admit(self.0.limits)?;
        let name = stage_name(plan.id());
        for target in [&name, &package_name(plan.id())] {
            ensure!(
                matches!(self.0.directory.symlink_metadata(target),Err(error) if error.kind()==std::io::ErrorKind::NotFound),
                "model '{}' already has a staged or published cache entry; resume or explicitly delete that entry",
                plan.id()
            );
        }
        let temporary = tempfile::Builder::new()
            .prefix(".nxrt-prepare-")
            .tempdir_in(&self.0.path)
            .context("create private transfer initialization directory")?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temporary.path().join(MANIFEST))?;
        file.write_all(&manifest)
            .context("write immutable transfer plan")?;
        file.sync_all().context("persist immutable transfer plan")?;
        drop(file);
        self.0
            .directory
            .rename(
                temporary
                    .path()
                    .file_name()
                    .expect("private temporary name"),
                &self.0.directory,
                &name,
            )
            .context("publish durable pending transfer")?;
        let _ = temporary.keep();
        *usage = total;
        Ok(PendingTransfer {
            stage: Stage {
                root: Arc::clone(&self.0),
                path: self.0.path.join(&name),
                directory: name,
                plan: Arc::new(plan),
                credit,
            },
        })
    }
    pub(crate) fn remove(&self, package: Arc<PublishedPackage>) -> Result<(), CacheOperationError> {
        if !Arc::ptr_eq(&self.0, &package.root) {
            return Err(CacheOperationError {
                cause: anyhow::anyhow!(
                    "model '{}' belongs to a different cache; remove it through its owning server",
                    package.id()
                ),
                package,
            });
        }
        let package=Arc::try_unwrap(package).map_err(|package|CacheOperationError{cause:anyhow::anyhow!("model '{}' still has active cache owners; finish or cancel active requests before deleting it",package.id()),package})?;
        self.0
            .remove_directory(&package.directory, &package.plan, package.usage)
            .map_err(|cause| CacheOperationError {
                package: Arc::new(package),
                cause,
            })
    }
}
impl CacheRoot {
    fn remove_directory(
        &self,
        name: &str,
        plan: &DownloadPlan,
        credit: Usage,
    ) -> anyhow::Result<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| anyhow::anyhow!("model cache accounting lock poisoned"))?;
        let mut nonce = [0_u8; 16];
        getrandom::fill(&mut nonce).context("allocate cache deletion transaction id")?;
        let trash = format!(".nxrt-trash-{:x}", Sha256::digest(nonce));
        self.directory
            .rename(name, &self.directory, &trash)
            .with_context(|| {
                format!(
                    "unpublish cached model '{}'; check permissions and active file handles",
                    plan.id()
                )
            })?;
        match self.purge(&trash, plan) {
            Ok(()) => usage.subtract(credit),
            Err(error) => {
                tracing::error!(model=plan.id(),%trash,%error,"model is unpublished; cleanup is pending and quota remains charged, repair cache permissions and restart to retry cleanup")
            }
        }
        Ok(())
    }
    fn purge(&self, name: &str, plan: &DownloadPlan) -> anyhow::Result<()> {
        let dir = self.directory.open_dir(name)?;
        verify_files(&dir, plan, true)?;
        let mut directories = Vec::new();
        let mut stack = vec![(dir.try_clone()?, String::new())];
        while let Some((parent, prefix)) = stack.pop() {
            for entry in parent.entries()? {
                let entry = entry?;
                let child = entry.file_name();
                let path = format!(
                    "{prefix}{}",
                    child.to_str().context("non-UTF-8 cache cleanup entry")?
                );
                if entry.file_type()?.is_dir() {
                    directories.push(path.clone());
                    stack.push((parent.open_dir(&child)?, format!("{path}/")));
                } else if path != MANIFEST {
                    let mut options = OpenOptions::new();
                    options.read(true).write(true);
                    #[cfg(unix)]
                    {
                        use cap_std::fs::OpenOptionsExt;
                        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
                    }
                    let file = dir.open_with(&path, &options)?.into_std();
                    file.lock().with_context(|| format!(
                        "wait for staged file {path:?} to finish its disk operation before deletion"
                    ))?;
                    drop(file);
                    dir.remove_file(&path)?;
                }
            }
        }
        directories.sort_by_key(|path| std::cmp::Reverse(path.len()));
        for path in directories {
            dir.remove_dir(path)?;
        }
        // Keep the plan until artifact cleanup succeeds, so an interrupted purge is recoverable.
        dir.remove_file(MANIFEST)?;
        self.directory.remove_dir(name)?;
        Ok(())
    }
}
fn verify_files(directory: &Dir, plan: &DownloadPlan, partial: bool) -> anyhow::Result<()> {
    let expected = plan
        .files()
        .map(|(path, bytes)| (path.to_owned(), bytes))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut found = std::collections::BTreeSet::new();
    let mut stack = vec![(directory.try_clone()?, String::new())];
    while let Some((dir, prefix)) = stack.pop() {
        for entry in dir.entries()? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF-8 cached package path"))?;
            let path = format!("{prefix}{name}");
            let kind = entry.file_type()?;
            if kind.is_dir() {
                let nested = format!("{path}/");
                ensure!(
                    expected.keys().any(|file| file.starts_with(&nested)),
                    "unexpected cache directory {path:?}; restore the immutable package"
                );
                stack.push((dir.open_dir(&name)?, nested));
            } else if path == MANIFEST {
                ensure!(kind.is_file(), "cache manifest is not a regular file");
            } else {
                let bytes = expected.get(&path).with_context(|| {
                    format!("unexpected cache file {path:?}; restore the immutable package")
                })?;
                ensure!(
                    kind.is_file(),
                    "cache file {path:?} is not regular; restore the immutable package"
                );
                let actual = entry.metadata()?.len();
                ensure!(
                    if partial {
                        actual <= *bytes
                    } else {
                        actual == *bytes
                    },
                    "cached model '{}' file {path:?} has invalid length {actual}, expected {}; cancel or redownload this entry",
                    plan.id(),
                    bytes
                );
                found.insert(path);
            }
        }
    }
    ensure!(
        partial || found.len() == expected.len(),
        "cached model '{}' is missing files; download the package again",
        plan.id()
    );
    Ok(())
}
impl Stage {
    fn publish(self) -> Result<Arc<PublishedPackage>, TransferFailure> {
        let result = (|| -> anyhow::Result<()> {
            if onnx_genai_ort::PipelineModelDirectory::load_if_declared(&self.path)?.is_none() {
                onnx_genai_ort::ModelDirectory::load(&self.path)
                    .context("validate downloaded model package before publication")?;
            }
            let dir = self.root.directory.open_dir(&self.directory)?;
            verify_files(&dir, &self.plan, false)?;
            let _usage = self
                .root
                .usage
                .lock()
                .map_err(|_| anyhow::anyhow!("model cache accounting lock poisoned"))?;
            let name = package_name(self.plan.id());
            ensure!(
                matches!(self.root.directory.symlink_metadata(&name),Err(error) if error.kind()==std::io::ErrorKind::NotFound),
                "model '{}' already has a published cache entry",
                self.plan.id()
            );
            self.root
                .directory
                .rename(&self.directory, &self.root.directory, name)
                .context("atomically publish verified model package")?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Arc::new(PublishedPackage {
                directory: package_name(self.plan.id()),
                root: self.root,
                plan: self.plan,
                usage: self.credit,
            })),
            Err(cause) => Err(TransferFailure {
                transfer: PendingTransfer { stage: self },
                cause,
            }),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model_hub::{
        HubModel,
        tests::{Transfer, hub},
    };
    pub(crate) async fn complete(
        cache: &ModelCache,
        client: &HubClient,
        plan: DownloadPlan,
    ) -> Arc<PublishedPackage> {
        let transfer = cache.begin(plan).unwrap();
        let (_control, mut pause) = tokio::sync::watch::channel(false);
        match transfer.run(client, &mut pause, |_| {}).await {
            TransferOutcome::Complete(complete) => {
                tokio::task::spawn_blocking(move || complete.publish())
                    .await
                    .unwrap()
                    .unwrap()
            }
            TransferOutcome::Failed(failure) => {
                panic!("fixture transfer failed: {:#}", failure.cause)
            }
            TransferOutcome::Paused(_) => panic!("fixture transfer unexpectedly paused"),
        }
    }
    fn limits() -> CacheLimits {
        CacheLimits {
            bytes: 4 * 1024 * 1024,
            files: 32,
            models: 2,
        }
    }
    async fn plan(client: &HubClient) -> DownloadPlan {
        client
            .plan(
                HubModel::new("owner/package".into(), "main".into()).unwrap(),
                HubLimits {
                    bytes: 1024 * 1024,
                    files: 16,
                },
            )
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn cache_publication_survives_restart_and_native_package_loading() {
        let root = tempfile::tempdir().unwrap();
        let (cache, packages, pending) = ModelCache::open(root.path(), limits()).unwrap();
        assert!(packages.is_empty());
        assert!(pending.is_empty());
        let (client, task) = hub(Transfer::Valid).await;
        let package = complete(&cache, &client, plan(&client).await).await;
        assert_eq!(package.id(), "owner/package");
        assert_eq!(package.commit(), "a".repeat(40));
        crate::AppState::load(&package.spec().path, Some(package.id().into())).unwrap();
        drop(package);
        drop(cache);
        let (cache, mut packages, pending) = ModelCache::open(root.path(), limits()).unwrap();
        assert_eq!(packages.len(), 1);
        assert!(pending.is_empty());
        let package = packages.pop().unwrap();
        assert_eq!(package.id(), "owner/package");
        let path = package.spec().path;
        cache.remove(package).unwrap();
        assert!(!path.exists());
        drop(cache);
        assert!(
            ModelCache::open(root.path(), limits())
                .unwrap()
                .1
                .is_empty()
        );
        task.abort();
    }
    #[tokio::test]
    async fn cache_reservations_bound_unpublished_downloads_and_release_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(
            root.path(),
            CacheLimits {
                models: 1,
                ..limits()
            },
        )
        .unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let plan = plan(&client).await;
        let stage = cache
            .begin(DownloadPlan::decode(&plan.encode().unwrap(), cache.hub_limits()).unwrap())
            .unwrap();
        assert!(!root.path().join(package_name(plan.id())).exists());
        let error = cache
            .begin(DownloadPlan::decode(&plan.encode().unwrap(), cache.hub_limits()).unwrap())
            .err()
            .expect("pending reservations count against quota");
        assert!(error.to_string().contains("quota exhausted"), "{error:#}");
        stage.cancel().unwrap();
        let stage = cache
            .begin(DownloadPlan::decode(&plan.encode().unwrap(), cache.hub_limits()).unwrap())
            .unwrap();
        stage.cancel().unwrap();
        assert_eq!(
            std::fs::read_dir(root.path()).unwrap().count(),
            1,
            "only process lock remains"
        );
        task.abort();
    }
    #[tokio::test]
    async fn cache_deletion_requires_matching_root_and_unique_package_ownership() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (other, _, _) = ModelCache::open(other.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let package = complete(&cache, &client, plan(&client).await).await;
        assert!(
            other
                .remove(Arc::clone(&package))
                .unwrap_err()
                .to_string()
                .contains("different cache")
        );
        let live = Arc::clone(&package);
        let failure = cache.remove(package).unwrap_err();
        assert!(failure.to_string().contains("active cache owners"));
        let package = failure.package;
        assert!(package.spec().path.is_dir());
        drop(live);
        cache.remove(package).unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn failed_transfers_preserve_the_plan_until_explicit_cancellation() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Corrupt).await;
        let transfer = cache.begin(plan(&client).await).unwrap();
        let (_control, mut pause) = tokio::sync::watch::channel(false);
        let TransferOutcome::Failed(failure) = transfer.run(&client, &mut pause, |_| {}).await
        else {
            panic!("corrupt transfer must fail")
        };
        assert!(failure.cause.to_string().contains("SHA-256"));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
        failure.transfer.cancel().unwrap();
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        task.abort();
    }
    #[test]
    fn cache_lock_and_unknown_entries_do_not_delete_operator_files() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let error = ModelCache::open(root.path(), limits())
            .err()
            .expect("cache has one process owner");
        assert!(error.to_string().contains("another server"), "{error:#}");
        drop(cache);
        std::fs::write(root.path().join("operator-data"), "keep").unwrap();
        assert!(ModelCache::open(root.path(), limits()).is_err());
        assert_eq!(
            std::fs::read_to_string(root.path().join("operator-data")).unwrap(),
            "keep"
        );
    }
    #[tokio::test]
    async fn detached_owner_threads_keep_package_files_until_their_state_is_destroyed() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let package = complete(&cache, &client, plan(&client).await).await;
        let (commands, _receiver) = tokio::sync::mpsc::channel(1);
        let (finish, finished) = std::sync::mpsc::channel::<()>();
        let (worker, ()) = crate::worker::WorkerHandle::spawn(
            crate::worker::WorkerId::PRIMARY,
            "cache-owner-test".into(),
            commands,
            move || Ok::<_, anyhow::Error>((finished, ())),
            |finished| {
                finished.recv().unwrap();
            },
        )
        .unwrap();
        let pool = crate::worker::WorkerPool::new(vec![worker]);
        pool.bind_cached_package(&package).unwrap();
        drop(pool); // the thread remains live even after its last handle is detached
        let failure = cache.remove(package).unwrap_err();
        assert!(failure.to_string().contains("active cache owners"));
        let package = failure.package;
        assert!(package.spec().path.is_dir());
        finish.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while Arc::strong_count(&package) > 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        cache.remove(package).unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn admitted_model_removal_finishes_after_its_waiter_is_cancelled() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let package = complete(&cache, &client, plan(&client).await).await;
        let path = package.spec().path;
        let registry = crate::registry::ModelRegistry::new_for_test();
        registry.register_cached(package, None).await.unwrap();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let blocked_cache = cache.clone();
        let blocker = tokio::task::spawn_blocking(move || {
            let _held = blocked_cache.0.usage.lock().unwrap();
            entered.send(()).unwrap();
            released.recv().unwrap();
        });
        ready.await.unwrap();
        let removing_registry = registry.clone();
        let removal = tokio::spawn(async move {
            removing_registry
                .remove_cached(cache, "owner/package".into())
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while registry.contains_available("owner/package").unwrap() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        removal.abort();
        release.send(()).unwrap();
        blocker.await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while path.exists() || registry.default_id().unwrap().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!registry.contains_available("owner/package").unwrap());
        task.abort();
    }
    #[tokio::test]
    async fn paused_transfers_survive_restart_and_keep_their_quota_reservation() {
        let root = tempfile::tempdir().unwrap();
        let bounds = CacheLimits {
            models: 1,
            ..limits()
        };
        let (cache, _, _) = ModelCache::open(root.path(), bounds).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let transfer = cache.begin(plan(&client).await).unwrap();
        let (control, mut pause) = tokio::sync::watch::channel(false);
        let outcome = transfer
            .run(&client, &mut pause, |state| {
                if state.package_bytes > 0 {
                    let _ = control.send(true);
                }
            })
            .await;
        let TransferOutcome::Paused(transfer) = outcome else {
            panic!("cooperative pause must preserve the transfer")
        };
        assert!(transfer.path().join("inference_metadata.yaml").is_file());
        drop(transfer);
        drop(cache);
        let (cache, published, mut pending) = ModelCache::open(root.path(), bounds).unwrap();
        assert!(published.is_empty());
        assert_eq!(pending.len(), 1);
        let transfer = pending.pop().unwrap();
        assert_eq!(transfer.plan().commit(), "a".repeat(40));
        let same =
            DownloadPlan::decode(&transfer.plan().encode().unwrap(), cache.hub_limits()).unwrap();
        assert!(
            cache
                .begin(same)
                .err()
                .unwrap()
                .to_string()
                .contains("quota exhausted")
        );
        let (_control, mut resume) = tokio::sync::watch::channel(false);
        let TransferOutcome::Complete(complete) = transfer.run(&client, &mut resume, |_| {}).await
        else {
            panic!("recovered transfer must complete")
        };
        let package = tokio::task::spawn_blocking(move || complete.publish())
            .await
            .unwrap()
            .unwrap();
        crate::AppState::load(&package.spec().path, Some(package.id().into())).unwrap();
        cache.remove(package).unwrap();
        task.abort();
    }
    #[tokio::test]
    async fn startup_finishes_interrupted_deletions_without_losing_the_manifest_early() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let package = complete(&cache, &client, plan(&client).await).await;
        cache
            .0
            .directory
            .rename(&package.directory, &cache.0.directory, ".nxrt-trash-test")
            .unwrap();
        cache
            .0
            .directory
            .open_dir(".nxrt-trash-test")
            .unwrap()
            .remove_file("tokenizer.json")
            .unwrap();
        drop(package);
        drop(cache);
        let (cache, published, pending) = ModelCache::open(
            root.path(),
            CacheLimits {
                bytes: 1,
                files: 1,
                models: 1,
            },
        )
        .unwrap();
        assert!(published.is_empty() && pending.is_empty());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        drop(cache);
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let package = complete(&cache, &client, plan(&client).await).await;
        cache.remove(package).unwrap();
        task.abort();
    }
    #[test]
    fn startup_recovers_only_private_metadata_initialization_directories() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".nxrt-prepare-test")).unwrap();
        std::fs::write(root.path().join(".nxrt-prepare-test").join(MANIFEST), b"{").unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        assert!(!root.path().join(".nxrt-prepare-test").exists());
        drop(cache);
        std::fs::create_dir(root.path().join(".nxrt-prepare-foreign")).unwrap();
        std::fs::write(
            root.path().join(".nxrt-prepare-foreign/operator-data"),
            "keep",
        )
        .unwrap();
        assert!(ModelCache::open(root.path(), limits()).is_err());
        assert_eq!(
            std::fs::read_to_string(root.path().join(".nxrt-prepare-foreign/operator-data"))
                .unwrap(),
            "keep"
        );
    }
    #[tokio::test]
    async fn owned_staged_files_reject_overlapping_disk_owners() {
        let root = tempfile::tempdir().unwrap();
        let (cache, _, _) = ModelCache::open(root.path(), limits()).unwrap();
        let (client, task) = hub(Transfer::Valid).await;
        let mut transfer = cache.begin(plan(&client).await).unwrap();
        let first = transfer.open_file("tokenizer.json").await.unwrap();
        let error = transfer.open_file("tokenizer.json").await.unwrap_err();
        assert!(error.to_string().contains("still owned"), "{error:#}");
        drop(first.into_std().await);
        drop(
            transfer
                .open_file("tokenizer.json")
                .await
                .unwrap()
                .into_std()
                .await,
        );
        transfer.cancel().unwrap();
        task.abort();
    }
}
