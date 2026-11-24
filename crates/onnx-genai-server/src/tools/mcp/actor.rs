//! One serial RPC lane per server on a node-owned runtime with joined teardown.
use super::*;
use rmcp::{
    ClientCacheConfig, RoleClient, ServiceExt,
    model::{ClientConfig, ClientRequest, Implementation, ProtocolVersion},
    service::{PeerRequestOptions, RunningService},
    transport::async_rw::AsyncRwTransport,
};
use std::process::Stdio;
use tokio::{
    process::{Child, Command},
    sync::{mpsc, oneshot, watch},
    task::JoinSet,
};

const CATALOG_BYTES: usize = 4 * 1024 * 1024;
type Running = RunningService<RoleClient, ClientConfig>;
pub(super) struct ConnectionCredit {
    pub(super) _lease: MemoryLease,
}
struct Retired {
    child: Child,
    _credit: Arc<ConnectionCredit>,
    _job: Option<Arc<Job>>,
    done: oneshot::Sender<()>,
}
struct ProcessOwner {
    child: Option<Child>,
    #[cfg(unix)]
    group: Option<nix::unistd::Pid>,
    credit: Arc<ConnectionCredit>,
    job: Option<Arc<Job>>,
    reaper: mpsc::UnboundedSender<Retired>,
}
impl ProcessOwner {
    fn retire(&mut self) -> Option<oneshot::Receiver<()>> {
        let mut child = self.child.take()?;
        #[cfg(unix)]
        if let Some(group) = self.group {
            let _ = nix::sys::signal::killpg(group, nix::sys::signal::Signal::SIGKILL);
        }
        let _ = child.start_kill();
        let (done, wait) = oneshot::channel();
        let retired = Retired {
            child,
            _credit: self.credit.clone(),
            _job: self.job.take(),
            done,
        };
        if self.reaper.send(retired).is_err() {
            tracing::error!(
                "MCP reaper stopped before its process owners; node worker lifecycle invariant violated"
            );
        }
        Some(wait)
    }
}
impl Drop for ProcessOwner {
    fn drop(&mut self) {
        self.retire();
    }
}
struct Session {
    running: Option<Running>,
    process: ProcessOwner,
    stderr: tokio::task::JoinHandle<()>,
    tail: Arc<std::sync::Mutex<Vec<u8>>>,
    fault: transport::Fault,
}
impl Session {
    async fn open(
        cfg: &config::ServerConfig,
        memory: &LedgerGovernor,
        reaper: mpsc::UnboundedSender<Retired>,
        job: Arc<Job>,
        deadline: Instant,
        stop: &mut watch::Receiver<bool>,
        abandoned: &oneshot::Sender<anyhow::Result<Reply>>,
    ) -> anyhow::Result<Self> {
        let lease = reserve(
            memory,
            cfg.frame_bytes * 128,
            "MCP connection and RPC staging",
        )?;
        let credit = Arc::new(ConnectionCredit { _lease: lease });
        let mut command = Command::new(&cfg.command);
        command
            .args(&cfg.args)
            .envs(&cfg.env)
            .current_dir(&cfg.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let child=command.spawn().with_context(||format!("start MCP server '{}' executable '{}' in '{}'; verify command installation, environment and directory",cfg.name,cfg.command,cfg.cwd.display()))?;
        #[cfg(unix)]
        let group = child
            .id()
            .and_then(|id| i32::try_from(id).ok())
            .map(nix::unistd::Pid::from_raw);
        let mut process = ProcessOwner {
            child: Some(child),
            #[cfg(unix)]
            group,
            credit: credit.clone(),
            job: Some(job.clone()),
            reaper,
        };
        let child = process
            .child
            .as_mut()
            .context("MCP child owner is unavailable")?;
        let input = child
            .stdin
            .take()
            .context("MCP child stdin was not piped")?;
        let output = child
            .stdout
            .take()
            .context("MCP child stdout was not piped")?;
        let mut errors = child
            .stderr
            .take()
            .context("MCP child stderr was not piped")?;
        // Drain the full stream but retain only a bounded diagnostic tail.
        let tail = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let stderr_tail = tail.clone();
        let stderr_credit = credit.clone();
        let mut stderr = tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let _credit = stderr_credit;
            let mut bytes = [0u8; 4096];
            loop {
                let n = match errors.read(&mut bytes).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                let mut tail = stderr_tail.lock().unwrap_or_else(|p| p.into_inner());
                let excess = tail.len().saturating_add(n).saturating_sub(4096);
                tail.drain(..excess);
                tail.extend_from_slice(&bytes[..n]);
            }
        });
        let fault = Arc::new(std::sync::Mutex::new(None));
        let transport = transport::CreditedTransport::new(
            AsyncRwTransport::<RoleClient, _, _>::new(output, input)
                .with_max_line_length(cfg.frame_bytes),
            memory.clone(),
            cfg.frame_bytes,
            fault.clone(),
            credit.clone(),
        );
        let mut client = ClientConfig::default();
        client.protocol_version = ProtocolVersion::V_2024_11_05;
        client.client_info = Implementation::new("nxrt", env!("CARGO_PKG_VERSION"));
        let startup = client.serve(transport);
        tokio::pin!(startup);
        let running = loop {
            tokio::select! {
                result=&mut startup => break result.with_context(||format!("initialize MCP server '{}'; verify its stdio JSON-RPC protocol and configured timeout",cfg.name)),
                _=tokio::time::sleep(Duration::from_millis(25))=>{
                    if job.check().is_err() || Instant::now()>=deadline || abandoned.is_closed() || *stop.borrow() {break Err(anyhow::anyhow!("MCP initialization cancelled or timed out; verify server startup and timeout_ms"));}
                },
                _=stop.changed()=>break Err(anyhow::anyhow!("MCP node is shutting down")),
            }
        };
        let running = match running {
            Ok(running) => running,
            Err(error) => {
                if let Some(wait) = process.retire() {
                    let _ = wait.await;
                }
                if tokio::time::timeout(Duration::from_millis(100), &mut stderr)
                    .await
                    .is_err()
                {
                    stderr.abort();
                    let _ = stderr.await;
                }
                let error = fault
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .unwrap_or(error);
                return Err(error.context(format!(
                    "MCP server '{}' startup stderr tail: {}",
                    cfg.name,
                    tail_text(&tail)
                )));
            }
        };
        running
            .peer()
            .set_response_cache_config(ClientCacheConfig::disabled())
            .await;
        process.job = None;
        Ok(Self {
            running: Some(running),
            process,
            stderr,
            tail,
            fault,
        })
    }
    async fn rpc(
        &mut self,
        request: ClientRequest,
        job: Arc<Job>,
        deadline: Instant,
        stop: &mut watch::Receiver<bool>,
        abandoned: &oneshot::Sender<anyhow::Result<Reply>>,
    ) -> anyhow::Result<Value> {
        self.process.job = Some(job.clone());
        let peer = self
            .running
            .as_ref()
            .context("MCP connection is closed; retry to restart the server")?
            .peer();
        let mut pending = peer
            .send_request_with_option(request, PeerRequestOptions::no_options())
            .await
            .context("send MCP request; verify the server remains alive")?;
        let result = loop {
            tokio::select! {
                result=&mut pending.rx=>break Some(result.context("MCP response channel closed; retry to restart the server")?),
                _=tokio::time::sleep(Duration::from_millis(25))=>{if job.check().is_err() || Instant::now()>=deadline || abandoned.is_closed() || *stop.borrow(){break None;}},
                _=stop.changed()=>break None,
            }
        };
        if let Some(result) = result {
            self.process.job = None;
            return serde_json::to_value(result.context("MCP server returned an RPC error")?)
                .context("encode MCP result; reduce tool output if host memory is exhausted");
        }
        // Never send cancellation for initialize. Here the SDK has issued a
        // normal RPC and its exact request id is owned by pending.
        let _ = tokio::time::timeout(
            Duration::from_millis(100),
            pending.cancel(Some("HTTP tool call cancelled or timed out".into())),
        )
        .await;
        // A normal ping is a bounded read barrier after the cancellation write.
        // The service gets a chance to consume the notification before teardown;
        // a broken/unresponsive peer still cannot retain the worker indefinitely.
        let _ = tokio::time::timeout(
            Duration::from_millis(100),
            peer.send_request(ClientRequest::PingRequest(Default::default())),
        )
        .await;
        anyhow::bail!(
            "MCP tool request cancelled or exceeded timeout_ms; its stdio session will be stopped before another call"
        )
    }
    async fn close(mut self) {
        let cleanup = self.running.take().map(|running| {
            tokio::spawn(async move {
                let _ = running.cancel().await;
            })
        });
        let mut cleanup = cleanup;
        let completed = if let Some(task) = cleanup.as_mut() {
            tokio::time::timeout(Duration::from_millis(100), task)
                .await
                .is_ok()
        } else {
            true
        };
        if let Some(wait) = self.process.retire() {
            let _ = wait.await;
        }
        if !completed && let Some(task) = cleanup {
            let _ = task.await;
        }
        self.stderr.abort();
        let _ = self.stderr.await;
    }
}
fn reserve(memory: &LedgerGovernor, bytes: usize, purpose: &str) -> anyhow::Result<MemoryLease> {
    memory.reserve(Tier::Host,u64::try_from(bytes)?,MemoryRole::Activation,HolderId::new(0))
        .with_context(||format!("reserve {purpose}; increase --tools-max-host-bytes, reduce MCP output, or wait for another tool call"))
}
pub(super) enum Operation {
    Discover,
    Call {
        name: String,
        arguments: serde_json::Map<String, Value>,
    },
}
pub(super) struct Reply {
    pub value: Value,
    pub credit: MemoryLease,
}
struct Request {
    operation: Operation,
    job: Arc<Job>,
    response: oneshot::Sender<anyhow::Result<Reply>>,
}
pub(super) struct Lane {
    pub name: String,
    pub timeout: Duration,
    tx: mpsc::Sender<Request>,
}
pub(super) struct Worker {
    pub lanes: Vec<Lane>,
    stop: watch::Sender<bool>,
    thread: Option<std::thread::JoinHandle<()>>,
    _config: MemoryLease,
}
impl Worker {
    pub fn start(config: Config, memory: LedgerGovernor, capacity: usize) -> anyhow::Result<Self> {
        let (stop, shutdown) = watch::channel(false);
        let mut lanes = Vec::new();
        let mut actors = Vec::new();
        for cfg in config.servers {
            let (tx, rx) = mpsc::channel(capacity);
            lanes.push(Lane {
                name: cfg.name.clone(),
                timeout: cfg.timeout,
                tx,
            });
            actors.push((cfg, rx));
        }
        let (ready, started) = std::sync::mpsc::sync_channel(1);
        let thread=std::thread::Builder::new().name("nxrt-mcp".into()).spawn(move||{
            let runtime=match tokio::runtime::Builder::new_current_thread().enable_all().build(){Ok(runtime)=>{let _=ready.send(Ok::<_,String>(()));runtime},Err(error)=>{let _=ready.send(Err(error.to_string()));return;}};
            runtime.block_on(async move {
                // At most one child per bounded configured lane. Each lane
                // awaits retirement before respawn; panic retirement adds at
                // most one child per terminating lane. This queue cannot grow
                // independently of that ownership bound.
                let (reaper,mut retired)=mpsc::unbounded_channel::<Retired>();
                let reap=tokio::spawn(async move{let mut pending=JoinSet::new();loop{tokio::select!{item=retired.recv()=>match item{Some(mut item)=>{pending.spawn(async move{let _=item.child.wait().await;let _=item.done.send(());});},None=>break},_=pending.join_next(),if !pending.is_empty()=>{}}}while pending.join_next().await.is_some(){}});
                let mut tasks=JoinSet::new();
                for(cfg,rx)in actors{tasks.spawn(run_lane(cfg,rx,memory.clone(),reaper.clone(),shutdown.clone()));}
                while let Some(result)=tasks.join_next().await {if let Err(error)=result{tracing::error!(%error,"MCP actor stopped unexpectedly");}}
                drop(reaper);let _=reap.await;
            });
        }).context("start MCP node worker; free thread/runtime resources and retry")?;
        match started
            .recv()
            .context("MCP node worker stopped during runtime initialization")?
        {
            Ok(()) => {}
            Err(error) => {
                let _ = thread.join();
                anyhow::bail!(
                    "create MCP node runtime: {error}; free host runtime resources and retry"
                );
            }
        }
        Ok(Self {
            lanes,
            stop,
            thread: Some(thread),
            _config: config._credit,
        })
    }
    pub async fn request(
        &self,
        index: usize,
        operation: Operation,
        job: Arc<Job>,
    ) -> anyhow::Result<Reply> {
        let lane = self
            .lanes
            .get(index)
            .context("MCP tool refers to an unavailable server lane; rediscover tools")?;
        let (response, result) = oneshot::channel();
        lane.tx
            .try_send(Request {
                operation,
                job,
                response,
            })
            .map_err(|_| {
                anyhow::anyhow!(
                    "MCP server '{}' is busy or stopped; retry after pending calls finish",
                    lane.name
                )
            })?;
        result.await.with_context(|| {
            format!(
                "MCP server '{}' worker stopped before returning a result; retry the call",
                lane.name
            )
        })?
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("MCP node worker panicked while shutting down");
        }
    }
}
async fn run_lane(
    cfg: config::ServerConfig,
    mut requests: mpsc::Receiver<Request>,
    memory: LedgerGovernor,
    reaper: mpsc::UnboundedSender<Retired>,
    mut stop: watch::Receiver<bool>,
) {
    let mut session: Option<Session> = None;
    let mut cooldown: Option<Instant> = None;
    loop {
        let request = tokio::select! {request=requests.recv()=>match request{Some(request)=>request,None=>break},_=stop.changed()=>break};
        if request.response.is_closed() {
            continue;
        }
        if let Err(error) = request.job.check() {
            let _=request.response.send(Err(error.context("MCP call expired while queued; reduce concurrency or increase the configured timeout_ms")));
            continue;
        }

        let discovery = matches!(request.operation, Operation::Discover);
        let duration = if discovery {
            cfg.timeout.min(Duration::from_secs(10))
        } else {
            cfg.timeout
        };
        let Some(deadline) = Instant::now().checked_add(duration) else {
            let _=request.response.send(Err(anyhow::anyhow!("MCP server '{}' timeout_ms exceeds this host's monotonic clock range; reduce the configured timeout",cfg.name)));
            continue;
        };
        let mut result=async{
            if cooldown.is_some_and(|until|Instant::now()<until){anyhow::bail!("MCP server '{}' is cooling down after a transport failure; retry in five seconds",cfg.name);}
            if session.as_ref().is_some_and(|s|s.running.as_ref().is_none_or(|r|r.is_closed())) && let Some(old)=session.take(){old.close().await;}
            if session.is_none(){session=Some(Session::open(&cfg,&memory,reaper.clone(),request.job.clone(),deadline,&mut stop,&request.response).await?);}
            let session=session.as_mut().context("MCP session initialization did not produce a client")?;
            match request.operation{
                Operation::Discover=>{
                    let mut credit=reserve(&memory,CATALOG_BYTES*32,"MCP tool catalog")?;
                    let mut definitions=Vec::new();let mut cursor=None;let mut cursors=std::collections::BTreeSet::new();let mut total=0usize;
                    loop{
                        let rpc=ClientRequest::ListToolsRequest(rmcp::model::ListToolsRequest::with_param(rmcp::model::PaginatedRequestParams::default().with_cursor(cursor)));
                        let value=session.rpc(rpc,request.job.clone(),deadline,&mut stop,&request.response).await?;
                        let tools=value.get("tools").and_then(Value::as_array).context("MCP tools/list result must contain a tools array; fix the configured server")?;
                        for tool in tools{
                            let name=tool.get("name").and_then(Value::as_str).context("MCP tool definition is missing its string name")?;
                            let schema=tool.get("inputSchema").filter(|v|v.is_object()).cloned().unwrap_or_else(||serde_json::json!({}));
                            let external=format!("{}_{}",cfg.name,name);
                            let definition=serde_json::json!({"tool":external,"display_name":external,"type":"mcp","uses_cwd":false,"permissions":{"write":false},"definition":{"type":"function","function":{"name":external,"description":tool.get("description").and_then(Value::as_str).unwrap_or(""),"parameters":schema}}});
                            total=total.checked_add(serde_json::to_vec(&definition)?.len()).context("MCP catalog size overflow")?;
                            ensure!(total<=CATALOG_BYTES && definitions.len()<512,"MCP tool catalog exceeds 4 MiB/512 tools; reduce the configured server's tool listing");
                            definitions.push(serde_json::json!({"name":name,"definition":definition}));
                        }
                        cursor=value.get("nextCursor").and_then(Value::as_str).map(str::to_owned);
                        if let Some(next)=&cursor{ensure!(cursors.len()<128 && cursors.insert(next.clone()),"MCP tools/list pagination repeats a cursor or exceeds 128 pages; fix the configured server");}else{break;}
                    }
                    let retained=total.saturating_mul(32).saturating_add(64*1024).min(CATALOG_BYTES*32);credit.shrink(credit.bytes()-retained as u64);
                    Ok(Reply{value:Value::Array(definitions),credit})
                },
                Operation::Call{name,arguments}=>{
                    let credit=reserve(&memory,cfg.frame_bytes*8,"MCP tool response ownership")?;
                    let rpc=ClientRequest::CallToolRequest(rmcp::model::CallToolRequest::new(rmcp::model::CallToolRequestParams::new(name).with_arguments(arguments)));
                    let value=session.rpc(rpc,request.job.clone(),deadline,&mut stop,&request.response).await?;
                    let mut text=String::new();
                    if let Some(content)=value.get("content").and_then(Value::as_array){for part in content{if part.get("type").and_then(Value::as_str)==Some("text"){if !text.is_empty(){text.push('\n');}text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));}}}
                    let value=if value.get("isError").and_then(Value::as_bool)==Some(true){serde_json::json!({"error":if text.is_empty(){"MCP tool returned an error".to_owned()}else{text}})}else{serde_json::json!({"plain_text_response":text})};
                    Ok(Reply{value,credit})
                }
            }
        }.await;
        let transport_failed = result.as_ref().err().is_some_and(|error| {
            !matches!(
                error.downcast_ref::<rmcp::ServiceError>(),
                Some(rmcp::ServiceError::McpError(_))
            )
        });
        if !transport_failed && let Some(session) = &mut session {
            session.process.job = None;
        }
        if discovery || transport_failed {
            if let Some(mut old) = session.take() {
                old.process.job = Some(request.job.clone());
                let tail = old.tail.clone();
                let fault = old.fault.clone();
                old.close().await;
                if let Err(error) = result {
                    let error = fault
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .take()
                        .unwrap_or(error);
                    result = Err(error.context(format!(
                        "MCP server '{}' stderr tail: {}",
                        cfg.name,
                        tail_text(&tail)
                    )));
                }
            }
            if transport_failed
                && result
                    .as_ref()
                    .err()
                    .is_none_or(|error| !error.is::<onnx_runtime_memory_api::MemoryError>())
            {
                cooldown = Some(Instant::now() + Duration::from_secs(5));
            }
        }
        let _ = request.response.send(result);
    }
    if let Some(old) = session {
        old.close().await;
    }
}

fn tail_text(tail: &std::sync::Mutex<Vec<u8>>) -> String {
    String::from_utf8_lossy(&tail.lock().unwrap_or_else(|p| p.into_inner())).into_owned()
}
