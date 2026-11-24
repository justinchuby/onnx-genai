//! MCP stdio tool client. Protocol lifecycle and correlation use the official SDK.
use super::*;
mod config;
mod transport;
use config::Config;
pub use config::McpSources;

mod actor;

pub(crate) struct Target {
    owner: Arc<Tools>,
    kind: TargetKind,
    controls: Controls,
}
enum TargetKind {
    Builtin(BuiltinTool),
    Mcp(Arc<Tool>),
}
#[derive(Default)]
struct Controls {
    cwd: Option<String>,
    response_type: Option<String>,
}
impl Target {
    pub(crate) fn streaming(&self) -> bool {
        matches!(
            self.kind,
            TargetKind::Builtin(BuiltinTool::ExecShellCommand)
        )
    }
    pub(crate) fn timeout(&self) -> Duration {
        match &self.kind {
            TargetKind::Builtin(BuiltinTool::FileGlobSearch | BuiltinTool::GrepSearch) => {
                Duration::from_secs(15)
            }
            TargetKind::Builtin(_) => TIMEOUT,
            TargetKind::Mcp(tool) => tool.timeout,
        }
    }
    pub(crate) fn frontend_controls(&mut self, cwd: Option<String>, response_type: Option<String>) {
        self.controls = Controls { cwd, response_type };
    }
    pub(crate) async fn invoke(
        self,
        params: &str,
        cwd: PathBuf,
        base64: bool,
        job: Arc<Job>,
        events: Option<tokio::sync::mpsc::Sender<Value>>,
    ) -> anyhow::Result<ToolValue> {
        match self.kind {
            TargetKind::Builtin(tool) => tokio::time::timeout_at(
                tokio::time::Instant::from_std(job.deadline()),
                self.owner.execute(tool, params, cwd, base64, job, events),
            )
            .await
            .context(
                "host-tool operation deadline exceeded; narrow the operation or reduce its output",
            )?,
            TargetKind::Mcp(tool) => {
                self.owner
                    .mcp
                    .as_ref()
                    .context("MCP client is not configured")?
                    .call(&tool, &self.controls, params, job)
                    .await
            }
        }
    }
}
pub(crate) struct Tool {
    lane: usize,
    name: String,
    external_name: String,
    pub definition: Value,
    timeout: Duration,
    _credit: Arc<MemoryLease>,
}
#[derive(Default)]
pub(crate) struct Catalog {
    pub tools: std::collections::BTreeMap<String, Arc<Tool>>,
    pub wire_bytes: usize,
}
pub(super) struct Client {
    worker: actor::Worker,
    memory: LedgerGovernor,
    response_budget: u64,
}
impl Client {
    pub fn new(
        sources: &McpSources,
        memory: LedgerGovernor,
        cwd: &Path,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let config = Config::load(sources, &memory, cwd)?;
        let response_budget = config
            ._credit
            .bytes()
            .saturating_add((sources.max_message_bytes.get() * 8) as u64);
        Ok(Self {
            worker: actor::Worker::start(config, memory.clone(), capacity)?,
            memory,
            response_budget,
        })
    }
    pub fn is_empty(&self) -> bool {
        self.worker.lanes.is_empty()
    }
    async fn discover(&self, tools: &Tools, job: Arc<Job>) -> anyhow::Result<Catalog> {
        let mut catalog = Catalog::default();
        for (index, lane) in self.worker.lanes.iter().enumerate() {
            let reply = match self
                .worker
                .request(index, actor::Operation::Discover, job.clone())
                .await
            {
                Ok(reply) => reply,
                Err(error) if error.is::<onnx_runtime_memory_api::MemoryError>() => {
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(server=%lane.name,error=%format!("{error:#}"),"MCP discovery failed; this server's tools are unavailable");
                    continue;
                }
            };
            let credit = Arc::new(reply.credit);
            for entry in reply
                .value
                .as_array()
                .context("MCP discovery returned a non-array tool catalog")?
            {
                let name = entry["name"]
                    .as_str()
                    .context("MCP discovery tool name is not a string")?;
                let definition = &entry["definition"];
                let external = definition["tool"]
                    .as_str()
                    .context("MCP discovery external name is not a string")?;
                if BuiltinTool::parse(external).is_some_and(|t| tools.enabled(t))
                    || catalog.tools.contains_key(external)
                {
                    tracing::warn!(tool=%external,"MCP tool name collides with an enabled tool; keeping the first definition");
                    continue;
                }
                catalog.wire_bytes = catalog
                    .wire_bytes
                    .checked_add(serde_json::to_vec(definition)?.len())
                    .context("combined MCP tool catalog size overflow")?;
                ensure!(
                    catalog.wire_bytes <= 4 * 1024 * 1024,
                    "combined MCP tool catalog exceeds 4 MiB; reduce configured tools"
                );
                catalog.tools.insert(
                    external.to_owned(),
                    Arc::new(Tool {
                        lane: index,
                        name: name.to_owned(),
                        external_name: external.to_owned(),
                        definition: definition.clone(),
                        timeout: lane.timeout,
                        _credit: credit.clone(),
                    }),
                );
            }
        }
        Ok(catalog)
    }
    async fn call(
        &self,
        tool: &Tool,
        controls: &Controls,
        params: &str,
        job: Arc<Job>,
    ) -> anyhow::Result<ToolValue> {
        let mut arguments: serde_json::Map<String, Value> = serde_json::from_str(params)
            .context("MCP tool params must be a JSON object matching its advertised schema")?;
        // Runtime controls are owned by the frontend, never model-selected.
        for key in ["cwd", "runtime", "resp_type"] {
            arguments.remove(key);
        }
        if let Some(cwd) = &controls.cwd {
            arguments.insert("cwd".into(), Value::String(cwd.clone()));
        }
        if let Some(kind) = &controls.response_type {
            arguments.insert("resp_type".into(), Value::String(kind.clone()));
        }
        let credit=self.memory.reserve(Tier::Host,self.response_budget,MemoryRole::Activation,HolderId::new(0)).context("reserve MCP response/error ownership; reduce output/concurrency or increase --tools-max-host-bytes")?;
        let reply = self
            .worker
            .request(
                tool.lane,
                actor::Operation::Call {
                    name: tool.name.clone(),
                    arguments,
                },
                job.clone(),
            )
            .await;
        match reply {
            Ok(reply) => Ok(ToolValue {
                value: reply.value,
                leases: vec![credit, reply.credit],
                job,
            }),
            Err(error) if error.is::<onnx_runtime_memory_api::MemoryError>() => Err(error),
            Err(error) => Ok(ToolValue {
                value: serde_json::json!({"error":format!("MCP tool '{}': {error:#}",tool.external_name)}),
                leases: vec![credit],
                job,
            }),
        }
    }
}
impl Tools {
    pub(crate) async fn catalog(&self, job: Arc<Job>) -> anyhow::Result<&Catalog> {
        self.catalog
            .get_or_try_init(|| async {
                match &self.mcp {
                    Some(client) => {
                        job.begin_operation(Duration::from_secs(
                            u64::try_from(client.worker.lanes.len())?
                                .saturating_mul(10)
                                .saturating_add(5),
                        ))?;
                        client.discover(self, job).await
                    }
                    None => Ok(Catalog::default()),
                }
            })
            .await
    }
    pub(crate) fn select_builtin(self: &Arc<Self>, tool: BuiltinTool) -> Option<Target> {
        self.enabled(tool).then(|| Target {
            owner: self.clone(),
            kind: TargetKind::Builtin(tool),
            controls: Controls::default(),
        })
    }
    pub(crate) async fn target(
        self: &Arc<Self>,
        name: &str,
        job: Arc<Job>,
    ) -> anyhow::Result<Option<Target>> {
        if let Some(target) = BuiltinTool::parse(name).and_then(|tool| self.select_builtin(tool)) {
            return Ok(Some(target));
        }
        Ok(self
            .catalog(job)
            .await?
            .tools
            .get(name)
            .cloned()
            .map(|tool| Target {
                owner: self.clone(),
                kind: TargetKind::Mcp(tool),
                controls: Controls::default(),
            }))
    }
}

#[cfg(test)]
mod tests;
