//! Bounded, cancellable process I/O. No shell is used for internal commands.
use super::*;
use std::process::Stdio;
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
    sync::mpsc::Sender,
};

pub(super) struct Output {
    pub output: Vec<u8>,
    pub stdout: Vec<u8>,
    pub code: i32,
    pub timed_out: bool,
    pub truncated: bool,
}
struct ChildGuard {
    child: Option<Child>,
    job: Arc<Job>,
    #[cfg(unix)]
    group: Option<nix::unistd::Pid>,
}
impl ChildGuard {
    fn stop(&mut self) {
        if let Some(child) = &mut self.child {
            #[cfg(unix)]
            if let Some(group) = self.group {
                let _ = nix::sys::signal::killpg(group, nix::sys::signal::Signal::SIGKILL);
            }
            let _ = child.start_kill();
        }
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.stop();
        if let Some(mut child) = self.child.take() {
            // Hold admission until the terminated process has actually been reaped.
            let job = self.job.clone();
            tokio::spawn(async move {
                let _ = child.wait().await;
                drop(job);
            });
        }
    }
}
#[derive(Default)]
struct Utf8Chunks {
    pending: Vec<u8>,
}
impl Utf8Chunks {
    fn push(&mut self, bytes: &[u8], finish: bool) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let end = error.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[..end]).expect("validated UTF-8 prefix"),
                    );
                    if let Some(length) = error.error_len() {
                        output.push('\u{fffd}');
                        self.pending.drain(..end + length);
                    } else {
                        self.pending.drain(..end);
                        if finish {
                            output.push('\u{fffd}');
                            self.pending.clear();
                        }
                        break;
                    }
                }
            }
        }
        output
    }
}
async fn emit(
    events: Option<&Sender<Value>>,
    chunk: String,
    deadline: Instant,
    job: &Job,
) -> anyhow::Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    if let Some(events) = events {
        let send = events.send(serde_json::json!({"chunk":chunk}));
        tokio::pin!(send);
        loop {
            tokio::select! {
                result=&mut send=>{result.context("tool stream consumer disconnected")?;break;},
                _=tokio::time::sleep(Duration::from_millis(25))=>{job.check()?;ensure!(Instant::now()<deadline,"tool stream exceeded its output deadline; read the response promptly or reduce command output");}
            }
        }
    }
    Ok(())
}
pub(super) async fn run(
    args: &[&str],
    cwd: &Path,
    limit: usize,
    timeout: Duration,
    job: &Arc<Job>,
    events: Option<&Sender<Value>>,
) -> anyhow::Result<Output> {
    job.check()?;
    let (program, args) = args
        .split_first()
        .context("tool process has no executable; check the tool configuration")?;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let child=command.spawn().with_context(||format!("start tool executable '{program}' in '{}'; verify the executable is installed and the working directory exists",cwd.display()))?;
    #[cfg(unix)]
    let group = child
        .id()
        .and_then(|id| i32::try_from(id).ok())
        .map(nix::unistd::Pid::from_raw);
    let mut guard = ChildGuard {
        child: Some(child),
        job: job.clone(),
        #[cfg(unix)]
        group,
    };
    let child = guard
        .child
        .as_mut()
        .context("tool process owner is unavailable")?;
    let mut stdout = child
        .stdout
        .take()
        .context("tool stdout pipe is unavailable")?;
    let mut stderr = child
        .stderr
        .take()
        .context("tool stderr pipe is unavailable")?;
    let deadline = (Instant::now() + timeout).min(job.deadline);
    let mut out = Vec::new();
    let mut stdout_bytes = Vec::new();
    stdout_bytes
        .try_reserve_exact(limit)
        .context("allocate bounded tool stdout; reduce output size")?;
    out.try_reserve_exact(limit)
        .context("allocate bounded tool output; free host memory or reduce max_output_size")?;
    let (mut stdout_done, mut stderr_done) = (false, false);
    let mut status = None;
    let mut timed_out = false;
    let mut truncated = false;
    let mut decoder = Utf8Chunks::default();
    let mut left = [0u8; 8192];
    let mut right = [0u8; 8192];
    while !stdout_done || !stderr_done || status.is_none() {
        job.check()?;
        if Instant::now() >= deadline {
            timed_out = true;
            break;
        }
        let bytes = tokio::select! {
            result=stdout.read(&mut left),if !stdout_done=>{let n=result.context("read tool stdout")?;if n==0{stdout_done=true;}let take=n.min(limit.saturating_sub(stdout_bytes.len())); stdout_bytes.extend_from_slice(&left[..take]); left[..n].to_vec()},
            result=stderr.read(&mut right),if !stderr_done=>{let n=result.context("read tool stderr")?;if n==0{stderr_done=true;}right[..n].to_vec()},
            result=child.wait(),if status.is_none()=>{status=Some(result.context("wait for tool process")?);Vec::new()},
            _=tokio::time::sleep(Duration::from_millis(25))=>Vec::new(),
        };
        let take = bytes.len().min(limit.saturating_sub(out.len()));
        truncated |= take < bytes.len();
        out.extend_from_slice(&bytes[..take]);
        emit(events, decoder.push(&bytes[..take], false), deadline, job).await?;
    }
    if timed_out {
        guard.stop();
    }
    if status.is_none() {
        status = Some(
            guard
                .child
                .as_mut()
                .context("tool process owner is unavailable")?
                .wait()
                .await
                .context("reap stopped tool process")?,
        );
    }
    // The process is terminal; keep no completed child handle in a drop reaper.
    guard.child.take();
    if !truncated {
        emit(events, decoder.push(&[], true), job.deadline, job).await?;
    }
    Ok(Output {
        output: out,
        stdout: stdout_bytes,
        code: status.and_then(|status| status.code()).unwrap_or(-1),
        timed_out,
        truncated,
    })
}

#[derive(Deserialize)]
struct ShellArgs {
    command: String,
    #[serde(default = "default_timeout")]
    timeout: u64,
    #[serde(default = "default_output")]
    max_output_size: usize,
}
fn default_timeout() -> u64 {
    10
}
fn default_output() -> usize {
    TEXT_BYTES
}
pub(super) async fn shell(
    _tools: Arc<Tools>,
    params: &str,
    cwd: PathBuf,
    job: Arc<Job>,
    events: Option<Sender<Value>>,
) -> anyhow::Result<ToolValue> {
    let params:ShellArgs=serde_json::from_str(params).context("invalid exec_shell_command parameters; supply command and optional positive timeout/max_output_size")?;
    ensure!(
        params.timeout > 0 && params.max_output_size > 0,
        "timeout and max_output_size must be positive"
    );
    let args = if cfg!(windows) {
        vec!["cmd", "/c", params.command.as_str()]
    } else {
        vec!["sh", "-c", params.command.as_str()]
    };
    let result = run(
        &args,
        &cwd,
        params.max_output_size.min(TEXT_BYTES),
        Duration::from_secs(params.timeout.min(60)),
        &job,
        events.as_ref(),
    )
    .await?;
    let mut tail = String::new();
    if result.truncated {
        tail.push_str("\n[output truncated]");
    }
    tail.push_str(&format!("\n[exit code: {}]", result.code));
    if result.timed_out {
        tail.push_str(" [exit due to timed out]");
    }
    let value = if events.is_some() {
        emit(events.as_ref(), tail, job.deadline, &job).await?;
        Value::Null
    } else {
        let mut decoder = Utf8Chunks::default();
        let mut text = decoder.push(&result.output, !result.truncated);
        text.push_str(&tail);
        serde_json::json!({"plain_text_response":text})
    };
    Ok(ToolValue {
        value,
        leases: Vec::new(),
        job,
    })
}
