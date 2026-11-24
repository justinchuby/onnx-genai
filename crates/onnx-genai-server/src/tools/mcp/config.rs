//! Cursor-compatible stdio server configuration, independent from host built-ins.
use super::*;
use std::collections::BTreeMap;

const CONFIG_BYTES: usize = 1024 * 1024;
const SERVER_LIMIT: usize = 32;

#[derive(Clone)]
pub struct McpSources {
    pub file: Option<PathBuf>,
    pub json: Option<Arc<str>>,
    pub max_message_bytes: std::num::NonZeroUsize,
}
impl Default for McpSources {
    fn default() -> Self {
        Self {
            file: None,
            json: None,
            max_message_bytes: std::num::NonZeroUsize::new(256 * 1024).unwrap(),
        }
    }
}
impl std::fmt::Debug for McpSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpSources")
            .field("file", &self.file)
            .field("json", &self.json.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}
impl McpSources {
    pub(crate) fn configured(&self) -> bool {
        self.file.is_some() || self.json.is_some()
    }
}
#[derive(Deserialize)]
struct Document {
    #[serde(default, rename = "mcpServers")]
    servers: BTreeMap<String, Value>,
}
#[derive(Deserialize)]
struct Entry {
    #[serde(default)]
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    cwd: String,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
}
fn default_timeout() -> u64 {
    30_000
}

pub(super) struct ServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: PathBuf,
    pub timeout: Duration,
    pub frame_bytes: usize,
}
pub(super) struct Config {
    pub servers: Vec<ServerConfig>,
    // Configuration strings and descriptors live no longer than this lease.
    pub _credit: MemoryLease,
}
fn parse(
    text: &str,
    label: &str,
    cwd: &Path,
    frame_bytes: usize,
) -> anyhow::Result<Vec<ServerConfig>> {
    ensure!(
        text.len() <= CONFIG_BYTES,
        "{label} exceeds the {CONFIG_BYTES} byte MCP configuration limit; split the configuration"
    );
    let document: Document = serde_json::from_str(text)
        .with_context(|| format!("parse {label}; supply a Cursor-compatible mcpServers object"))?;
    ensure!(
        document.servers.len() <= SERVER_LIMIT,
        "{label} configures more than {SERVER_LIMIT} MCP servers; reduce the configured server count"
    );
    let mut result = Vec::new();
    for (name, value) in document.servers {
        let entry: Entry = serde_json::from_value(value)
            .with_context(|| format!("parse MCP server '{name}' in {label}; command/cwd must be strings, args a string array, env a string map and timeout_ms a positive integer"))?;
        if entry.command.is_empty() {
            tracing::warn!(server=%name, "MCP entry has no stdio command; skipping it");
            continue;
        }
        ensure!(
            entry.timeout_ms > 0 && entry.timeout_ms <= i32::MAX as u64,
            "MCP server '{name}' timeout_ms must be between 1 and 2147483647; use 30000 for the default call timeout"
        );
        ensure!(
            !entry.command.contains('\0')
                && entry.args.iter().all(|a| !a.contains('\0'))
                && !entry.cwd.contains('\0'),
            "MCP server '{name}' command/args/cwd contain a NUL byte; remove it before starting the server"
        );
        for (key, value) in &entry.env {
            ensure!(
                !key.is_empty() && !key.contains(['=', '\0']) && !value.contains('\0'),
                "MCP server '{name}' has an invalid environment entry name or NUL value; use nonempty names without '=' or NUL and values without NUL"
            );
        }
        let path = PathBuf::from(&entry.cwd);
        result.push(ServerConfig {
            name,
            command: entry.command,
            args: entry.args,
            env: entry.env,
            cwd: if path.is_absolute() {
                path
            } else {
                cwd.join(path)
            },
            timeout: Duration::from_millis(entry.timeout_ms),
            frame_bytes,
        });
    }
    Ok(result)
}
impl Config {
    pub(super) fn load(
        sources: &McpSources,
        memory: &LedgerGovernor,
        cwd: &Path,
    ) -> anyhow::Result<Self> {
        ensure!(
            sources.max_message_bytes.get() <= 8 * 1024 * 1024,
            "--mcp-max-message-bytes must be at most 8388608 (the upstream 8 MiB stdio line limit); reduce the configured limit"
        );
        use std::io::Read as _;
        let opened = sources
            .file
            .as_ref()
            .map(|path| -> anyhow::Result<_> {
                let file = files::regular(path)
                    .with_context(|| format!("open MCP configuration file '{}'", path.display()))?;
                let size = usize::try_from(
                    file.metadata()
                        .with_context(|| {
                            format!("inspect MCP configuration file '{}'", path.display())
                        })?
                        .len(),
                )
                .context("MCP configuration file exceeds this host's address space")?;
                ensure!(
                    size <= CONFIG_BYTES,
                    "MCP configuration file '{}' exceeds {CONFIG_BYTES} bytes; reduce its size",
                    path.display()
                );
                Ok((path, file, size))
            })
            .transpose()?;
        let size = opened
            .as_ref()
            .map_or(0, |(_, _, size)| *size)
            .checked_add(sources.json.as_ref().map_or(0, |text| text.len()))
            .context("combined MCP configuration size overflow")?;
        ensure!(
            size <= CONFIG_BYTES,
            "combined MCP configuration exceeds {CONFIG_BYTES} bytes; reduce file/inline configuration size"
        );
        let reserved = size
            .checked_mul(32)
            .and_then(|n| n.checked_add(64 * 1024))
            .context("MCP configuration staging size overflow")?;
        let mut credit=memory.reserve(Tier::Host,u64::try_from(reserved)?,MemoryRole::Activation,HolderId::new(0)).context("reserve MCP configuration staging; increase --tools-max-host-bytes or reduce configuration size")?;
        let file_text=opened.map(|(path,file,size)|->anyhow::Result<_>{
            let mut bytes=Vec::new();bytes.try_reserve_exact(size.saturating_add(1)).context("allocate MCP configuration read buffer; free host memory or reduce configuration size")?;
            file.take(u64::try_from(size)?.saturating_add(1)).read_to_end(&mut bytes).with_context(||format!("read MCP configuration file '{}'",path.display()))?;
            ensure!(bytes.len()<=size,"MCP configuration file '{}' grew beyond its admitted size; retry after the writer finishes",path.display());
            Ok((format!("MCP configuration file '{}'",path.display()),String::from_utf8(bytes).context("MCP configuration file must contain UTF-8 JSON")?))
        }).transpose()?;
        let mut servers = Vec::<ServerConfig>::new();
        let mut bytes = 0usize;
        for (label, text) in [
            file_text,
            sources
                .json
                .as_ref()
                .map(|text| ("inline --mcp-servers-json".to_owned(), text.to_string())),
        ]
        .into_iter()
        .flatten()
        {
            bytes = bytes
                .checked_add(text.len())
                .context("MCP configuration size overflow; use smaller configuration sources")?;
            ensure!(
                bytes <= CONFIG_BYTES,
                "combined MCP configuration exceeds {CONFIG_BYTES} bytes; reduce file/inline configuration size"
            );
            for server in parse(&text, &label, cwd, sources.max_message_bytes.get())? {
                if servers.iter().any(|existing| existing.name == server.name) {
                    tracing::warn!(server=%server.name, "duplicate MCP server name; keeping the first configuration");
                } else {
                    servers.push(server);
                }
            }
        }
        ensure!(
            servers.len() <= SERVER_LIMIT,
            "combined MCP configuration exceeds {SERVER_LIMIT} servers; reduce the configured server count"
        );
        let retained = bytes
            .saturating_mul(16)
            .saturating_add(64 * 1024)
            .min(reserved);
        credit.shrink(credit.bytes() - retained as u64);
        Ok(Self {
            servers,
            _credit: credit,
        })
    }
}
