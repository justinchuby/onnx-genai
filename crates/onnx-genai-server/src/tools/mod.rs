//! Explicitly enabled host tools used by the official llama.cpp frontend.
const REQUEST_EXPANSION: usize = 16;
mod files;
pub(crate) mod mcp;
pub use mcp::McpSources;
mod process;
#[cfg(test)]
mod tests;

use anyhow::{Context, ensure};
use onnx_runtime_memory_api::{MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryLease,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const TEXT_BYTES: usize = 16 * 1024;
const FILE_BYTES: usize = 32 * 1024 * 1024;
const LIST_BYTES: usize = 8 * 1024 * 1024;
const ENTRY_LIMIT: usize = 65536;
const TIMEOUT: Duration = Duration::from_secs(60);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum BuiltinTool {
    All,
    ReadFile,
    FileGlobSearch,
    GrepSearch,
    ExecShellCommand,
    WriteFile,
    EditFile,
    GetInfo,
}
impl BuiltinTool {
    const TOOLS: [Self; 7] = [
        Self::ReadFile,
        Self::FileGlobSearch,
        Self::GrepSearch,
        Self::ExecShellCommand,
        Self::WriteFile,
        Self::EditFile,
        Self::GetInfo,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::ReadFile => "read_file",
            Self::FileGlobSearch => "file_glob_search",
            Self::GrepSearch => "grep_search",
            Self::ExecShellCommand => "exec_shell_command",
            Self::WriteFile => "write_file",
            Self::EditFile => "edit_file",
            Self::GetInfo => "get_info",
        }
    }
    pub(crate) fn parse(name: &str) -> Option<Self> {
        Self::TOOLS.into_iter().find(|t| t.name() == name)
    }
}

/// These limits bound this node's host-tool I/O, independently of model execution.
#[derive(Debug, Clone)]
pub struct ToolsConfig {
    pub enabled: Vec<BuiltinTool>,
    pub max_concurrent: std::num::NonZeroUsize,
    pub max_host_bytes: std::num::NonZeroU64,
    pub allow_origins: Vec<String>,
    pub mcp: McpSources,
}
impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: Vec::new(),
            max_concurrent: std::num::NonZeroUsize::new(8).unwrap(),
            max_host_bytes: std::num::NonZeroU64::new(256 * 1024 * 1024).unwrap(),
            allow_origins: Vec::new(),
            mcp: McpSources::default(),
        }
    }
}
impl ToolsConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            (self.enabled.is_empty() && !self.mcp.configured())
                || self.max_host_bytes.get() >= 256 * 1024,
            "--tools-max-host-bytes must be at least 262144 bytes for tool control/response overhead; increase the budget"
        );
        ensure!(
            self.max_concurrent.get() <= Semaphore::MAX_PERMITS,
            "tools max_concurrent exceeds this platform's semaphore capacity; choose at most {}",
            Semaphore::MAX_PERMITS
        );
        for origin in &self.allow_origins {
            let url = reqwest::Url::parse(origin).with_context(|| {
                format!("invalid tools origin '{origin}'; use an http(s) origin without a path")
            })?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.path() == "/"
                    && url.query().is_none()
                    && url.fragment().is_none()
                    && url.origin().ascii_serialization() == *origin,
                "tools origin '{origin}' must be an exact http(s) origin without credentials, path or trailing slash"
            );
        }
        Ok(())
    }
}

pub(crate) struct Tools {
    enabled: BTreeSet<BuiltinTool>,
    slots: Arc<Semaphore>,
    memory: LedgerGovernor,
    cwd: PathBuf,
    home: Option<PathBuf>,
    origins: Vec<String>,
    writes: std::sync::Mutex<()>,
    mcp: Option<mcp::Client>,
    catalog: tokio::sync::OnceCell<mcp::Catalog>,
}
impl std::fmt::Debug for Tools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tools")
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}
impl Tools {
    pub(crate) fn new(config: &ToolsConfig) -> anyhow::Result<Option<Arc<Self>>> {
        config.validate()?;
        if config.enabled.is_empty() && !config.mcp.configured() {
            return Ok(None);
        }
        let enabled: BTreeSet<BuiltinTool> = if config.enabled.contains(&BuiltinTool::All) {
            BuiltinTool::TOOLS.into_iter().collect()
        } else {
            config.enabled.iter().copied().collect()
        };
        let memory = LedgerGovernor::new(LeaseLedger::new(0, config.max_host_bytes.get(), 0));
        let cwd = std::env::current_dir().context(
            "resolve host-tools working directory; start the server from an accessible directory",
        )?;
        let mcp = if config.mcp.configured() {
            Some(mcp::Client::new(
                &config.mcp,
                memory.clone(),
                &cwd,
                config.max_concurrent.get(),
            )?)
        } else {
            None
        };
        if enabled.is_empty() && mcp.as_ref().is_none_or(|client| client.is_empty()) {
            return Ok(None);
        }
        Ok(Some(Arc::new(Self {
            enabled,
            slots: Arc::new(Semaphore::new(config.max_concurrent.get())),
            memory,
            cwd,
            home: std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
                .map(PathBuf::from),
            origins: config.allow_origins.clone(),
            writes: std::sync::Mutex::new(()),
            mcp,
            catalog: tokio::sync::OnceCell::new(),
        })))
    }
    pub(crate) fn origin_allowed(&self, origin: &str) -> bool {
        if self.origins.iter().any(|o| o == origin) {
            return true;
        }
        reqwest::Url::parse(origin).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.username().is_empty()
                && url.password().is_none()
                && url.path() == "/"
                && url.query().is_none()
                && url.fragment().is_none()
                && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
        })
    }
    pub(crate) fn enabled(&self, tool: BuiltinTool) -> bool {
        self.enabled.contains(&tool)
    }
    pub(crate) fn definitions(&self) -> Value {
        let all: Value = serde_json::from_str(include_str!("definitions.json"))
            .expect("embedded tool definitions are valid JSON");
        Value::Array(
            all.as_array()
                .expect("tool definitions are an array")
                .iter()
                .filter(|item| {
                    BuiltinTool::parse(item["tool"].as_str().unwrap_or_default())
                        .is_some_and(|t| self.enabled(t))
                })
                .cloned()
                .collect(),
        )
    }
    pub(crate) fn request_bound(&self, maximum: usize) -> usize {
        let bytes = self
            .memory
            .ledger()
            .limit(Tier::Host)
            .saturating_sub(self.memory.ledger().used(Tier::Host))
            .saturating_sub(256 * 1024)
            / REQUEST_EXPANSION as u64;
        maximum.min(usize::try_from(bytes).unwrap_or(usize::MAX))
    }
    pub(crate) fn admit(&self, body_bytes: usize) -> anyhow::Result<Job> {
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .context("host tools are busy; retry after another tool call completes")?;
        let bytes = body_bytes
            .checked_mul(REQUEST_EXPANSION)
            .and_then(|n| n.checked_add(256 * 1024))
            .context("tool request allocation size overflow; send a smaller request")?;
        let lease = self.reserve(bytes)?;
        Ok(Job {
            _permit: permit,
            _lease: lease,
            cancelled: AtomicBool::new(false),
            deadline: std::sync::Mutex::new(Instant::now() + TIMEOUT),
        })
    }
    pub(crate) fn reserve(&self, bytes: usize) -> anyhow::Result<MemoryLease> {
        self.memory.reserve(Tier::Host,u64::try_from(bytes)?,MemoryRole::Activation,HolderId::new(0)).context("reserve host-tool I/O memory; retry after another call completes, reduce its size, or increase --tools-max-host-bytes")
    }
    pub(crate) fn cwd(&self, header: Option<&str>) -> PathBuf {
        header
            .filter(|v| !v.is_empty())
            .map_or_else(|| self.cwd.clone(), |path| self.resolve(&self.cwd, path))
    }
    fn resolve(&self, cwd: &Path, path: &str) -> PathBuf {
        let path = if path == "~" || path.starts_with("~/") {
            self.home.as_ref().map_or_else(
                || PathBuf::from(path),
                |home| home.join(path.strip_prefix("~/").unwrap_or("")),
            )
        } else {
            PathBuf::from(path)
        };
        let full = if path.is_absolute() {
            path
        } else {
            cwd.join(path)
        };
        let mut result = PathBuf::new();
        for part in full.components() {
            match part {
                Component::CurDir => {}
                Component::ParentDir => {
                    result.pop();
                }
                _ => result.push(part.as_os_str()),
            }
        }
        result
    }
    async fn execute(
        self: Arc<Self>,
        tool: BuiltinTool,
        params: &str,
        cwd: PathBuf,
        base64: bool,
        job: Arc<Job>,
        events: Option<tokio::sync::mpsc::Sender<Value>>,
    ) -> anyhow::Result<ToolValue> {
        job.check()?;
        if tool == BuiltinTool::ExecShellCommand {
            return process::shell(self, params, cwd, job, events).await;
        }
        if tool == BuiltinTool::GetInfo {
            let args = if cfg!(windows) {
                vec!["cmd", "/c", "ver"]
            } else {
                vec!["uname", "-a"]
            };
            let result = process::run(&args, &cwd, 4096, Duration::from_secs(5), &job, None).await;
            job.check()?;
            let os = match result {
                Ok(result) if result.code == 0 && !result.timed_out => {
                    String::from_utf8_lossy(&result.stdout).trim().to_owned()
                }
                _ => "unknown".to_owned(),
            };
            return Ok(ToolValue {
                value: serde_json::json!({"os":os,"cwd":cwd}),
                leases: Vec::new(),
                job,
            });
        }
        let git = if matches!(tool, BuiltinTool::FileGlobSearch | BuiltinTool::GrepSearch) {
            let input: files::ListingRequest = serde_json::from_str(params).context(
                "invalid file-search parameters; supply path and the tool's declared fields",
            )?;
            if tool == BuiltinTool::GrepSearch || input.kind.as_deref().unwrap_or("file") == "file"
            {
                let base = self.resolve(&cwd, &input.path);
                let lease = self.reserve(LIST_BYTES * 8)?;
                let result = process::run(
                    &[
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-C",
                        base.to_str().context(
                            "tool path is not valid Unicode; choose a Unicode directory",
                        )?,
                        "ls-files",
                        "-z",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "--",
                        ".",
                    ],
                    &cwd,
                    LIST_BYTES,
                    Duration::from_secs(15),
                    &job,
                    None,
                )
                .await;
                Some((result.ok().filter(|r| r.code == 0 && !r.timed_out), lease))
            } else {
                None
            }
        } else {
            None
        };
        let params = params.to_owned();
        let worker_job = job.clone();
        tokio::task::spawn_blocking(move || {
            files::invoke(&self, tool, &params, &cwd, base64, git, worker_job)
        })
        .await
        .context("host-tool worker stopped; inspect server logs and retry")?
    }
}

pub(crate) struct Job {
    _permit: OwnedSemaphorePermit,
    _lease: MemoryLease,
    cancelled: AtomicBool,
    deadline: std::sync::Mutex<Instant>,
}
impl Job {
    pub(crate) fn retain_body_credit(&mut self, actual: usize) -> anyhow::Result<()> {
        let keep = actual
            .checked_mul(REQUEST_EXPANSION)
            .and_then(|n| n.checked_add(256 * 1024))
            .context("tool body allocation size overflow")?;
        let keep = u64::try_from(keep)?;
        ensure!(
            keep <= self._lease.bytes(),
            "tool body exceeded its admitted byte limit; send a smaller request"
        );
        self._lease.shrink(self._lease.bytes() - keep);
        Ok(())
    }

    fn deadline(&self) -> Instant {
        *self
            .deadline
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
    pub(crate) fn begin_operation(&self, limit: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now()
            .checked_add(limit)
            .context("tool timeout exceeds this host's monotonic clock range; reduce timeout_ms")?;
        *self
            .deadline
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = deadline;
        Ok(())
    }
    fn check(&self) -> anyhow::Result<()> {
        ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "tool call cancelled by client disconnect"
        );
        ensure!(
            Instant::now() < self.deadline(),
            "tool call exceeded its operation deadline; narrow the operation and retry"
        );
        Ok(())
    }
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
pub(crate) struct ToolValue {
    pub value: Value,
    pub leases: Vec<MemoryLease>,
    pub job: Arc<Job>,
}
pub(crate) struct ResponseBytes {
    pub bytes: Vec<u8>,
    pub _leases: Vec<MemoryLease>,
    pub job: Arc<Job>,
}
impl AsRef<[u8]> for ResponseBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for ResponseBytes {
    fn drop(&mut self) {
        self.job.cancel();
    }
}
