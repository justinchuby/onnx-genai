use super::*;
use base64::Engine as _;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
};

#[derive(Deserialize)]
pub(super) struct ListingRequest {
    pub path: String,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
}
#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    #[serde(default = "one")]
    start_line: i64,
    #[serde(default = "unlimited")]
    end_line: i64,
    #[serde(default)]
    append_loc: bool,
}
fn one() -> i64 {
    1
}
fn unlimited() -> i64 {
    -1
}
#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}
#[derive(Deserialize)]
struct EditArgs {
    path: String,
    edits: Vec<Edit>,
}
#[derive(Deserialize)]
struct Edit {
    old_text: String,
    new_text: String,
}
#[derive(Deserialize)]
struct SearchArgs {
    path: String,
    #[serde(default = "all_pattern")]
    include: String,
    #[serde(default)]
    exclude: String,
    #[serde(default = "file_kind", rename = "type")]
    kind: String,
    #[serde(default)]
    max_depth: usize,
    #[serde(default = "hundred")]
    limit: usize,
}
fn all_pattern() -> String {
    "**".into()
}
fn file_kind() -> String {
    "file".into()
}
fn hundred() -> usize {
    100
}
#[derive(Deserialize)]
struct GrepArgs {
    path: String,
    pattern: String,
    #[serde(default = "all_pattern")]
    include: String,
    #[serde(default)]
    exclude: String,
    #[serde(default)]
    return_line_numbers: bool,
    #[serde(default)]
    literal: bool,
    #[serde(default)]
    ignore_case: bool,
    #[serde(default)]
    context_lines: usize,
}

fn regular(path: &Path) -> anyhow::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).with_context(|| {
        format!(
            "open tool file '{}'; verify the path and read permissions",
            path.display()
        )
    })?;
    ensure!(
        file.metadata()
            .with_context(|| format!(
                "inspect tool file metadata for '{}'; verify the file remains accessible",
                path.display()
            ))?
            .is_file(),
        "tool path '{}' is not a regular file; choose a readable regular file",
        path.display()
    );
    Ok(file)
}
fn read_bounded(path: &Path, limit: usize) -> anyhow::Result<Vec<u8>> {
    let file = regular(path)?;
    let length = usize::try_from(
        file.metadata()
            .with_context(|| {
                format!(
                    "inspect tool file metadata for '{}'; verify the file remains accessible",
                    path.display()
                )
            })?
            .len(),
    )
    .context("file size exceeds this host's address space")?;
    ensure!(
        length <= limit,
        "file '{}' exceeds the {limit} byte tool input limit; select a smaller file",
        path.display()
    );
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length.saturating_add(1))
        .context("allocate bounded file input; free host memory and retry")?;
    file.take(u64::try_from(limit)?.saturating_add(1))
        .read_to_end(&mut bytes).with_context(|| format!("read tool file '{}'; verify read permissions and retry after concurrent writes finish", path.display()))?;
    ensure!(
        bytes.len() <= limit,
        "file '{}' grew beyond the {limit} byte limit while reading; retry after the writer completes",
        path.display()
    );
    Ok(bytes)
}
fn text_read(
    tools: &Tools,
    args: ReadArgs,
    cwd: &Path,
    as_base64: bool,
    job: &Job,
) -> anyhow::Result<(Value, Vec<MemoryLease>)> {
    let path = tools.resolve(cwd, &args.path);
    let file = regular(&path)?;
    let length = usize::try_from(
        file.metadata()
            .with_context(|| {
                format!(
                    "inspect tool file metadata for '{}'; verify the file remains accessible",
                    path.display()
                )
            })?
            .len(),
    )?;
    if as_base64 {
        ensure!(
            length <= FILE_BYTES,
            "file '{}' exceeds the {FILE_BYTES} byte base64 limit; select a smaller file",
            path.display()
        );
        let lease = tools.reserve(
            length
                .checked_mul(6)
                .context("base64 response size overflow")?,
        )?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length.saturating_add(1))?;
        file.take(u64::try_from(length)?.saturating_add(1))
            .read_to_end(&mut bytes).with_context(|| format!("read tool file '{}'; verify read permissions and retry after concurrent writes finish", path.display()))?;
        ensure!(
            bytes.len() <= length,
            "file '{}' grew while reading; retry after the writer completes",
            path.display()
        );
        job.check()?;
        let mut encoded = String::new();
        encoded.try_reserve_exact(
            length
                .checked_add(2)
                .and_then(|n| n.checked_div(3))
                .and_then(|n| n.checked_mul(4))
                .context("base64 response size overflow")?,
        )?;
        base64::engine::general_purpose::STANDARD.encode_string(&bytes, &mut encoded);
        return Ok((
            serde_json::json!({"base64":encoded,"size_bytes":bytes.len()}),
            vec![lease],
        ));
    }
    ensure!(
        length <= TEXT_BYTES || args.end_line != -1,
        "file too large ({length} bytes, max {TEXT_BYTES}); use start_line/end_line to read a portion"
    );
    let mut reader = BufReader::new(file);
    let mut result = String::new();
    let mut line = Vec::new();
    let mut line_no = 1_i64;
    let mut truncated = false;
    loop {
        job.check()?;
        if args.end_line != -1 && line_no > args.end_line {
            break;
        }
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !line.is_empty() {
                append_line(&mut result, &line, line_no, &args, &mut truncated);
            }
            break;
        }
        let end = buffer.iter().position(|b| *b == b'\n');
        let count = end.map_or(buffer.len(), |p| p + 1);
        if line_no >= args.start_line {
            let payload = &buffer[..end.unwrap_or(count)];
            if line.len().saturating_add(payload.len()) > TEXT_BYTES {
                truncated = true;
                break;
            }
            line.extend_from_slice(payload);
        }
        reader.consume(count);
        if end.is_some() {
            if line_no >= args.start_line {
                append_line(&mut result, &line, line_no, &args, &mut truncated);
                line.clear();
            }
            line_no = line_no.saturating_add(1);
            if truncated {
                break;
            }
        }
    }
    if truncated {
        result.push_str("[output truncated]");
    }
    Ok((
        serde_json::json!({"plain_text_response":result}),
        Vec::new(),
    ))
}
fn append_line(
    output: &mut String,
    line: &[u8],
    number: i64,
    args: &ReadArgs,
    truncated: &mut bool,
) {
    let line = String::from_utf8_lossy(line);
    let prefix = if args.append_loc {
        format!("{number}→")
    } else {
        String::new()
    };
    if output
        .len()
        .saturating_add(prefix.len())
        .saturating_add(line.len())
        .saturating_add(1)
        > TEXT_BYTES
    {
        *truncated = true;
        return;
    }
    output.push_str(&prefix);
    output.push_str(&line);
    output.push('\n');
}

fn mutation_lock<'a>(tools: &'a Tools, job: &Job) -> anyhow::Result<std::sync::MutexGuard<'a, ()>> {
    loop {
        job.check()?;
        match tools.writes.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(std::sync::TryLockError::Poisoned(_)) => anyhow::bail!(
                "host-tool file transaction lock is poisoned; restart the server after inspecting the prior failure"
            ),
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(5))
            }
        }
    }
}
fn atomic_write(path: &Path, content: &[u8], job: &Job) -> anyhow::Result<()> {
    job.check()?;
    let destination = if path.exists() {
        ensure!(
            path.is_file(),
            "write target '{}' is not a regular file",
            path.display()
        );
        path.canonicalize()?
    } else {
        path.to_path_buf()
    };
    let parent = destination
        .parent()
        .context("write path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent directory '{}'", parent.display()))?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("stage tool write beside '{}'", destination.display()))?;
    if let Ok(metadata) = fs::metadata(&destination) {
        staged.as_file().set_permissions(metadata.permissions())?;
    }
    staged
        .write_all(content)
        .with_context(|| format!("write staged file '{}'", destination.display()))?;
    staged.as_file().sync_all()?;
    job.check()?;
    staged
        .persist(&destination)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "publish tool file '{}'; verify directory permissions and retry",
                destination.display()
            )
        })?;
    Ok(())
}

#[derive(Debug)]
struct Entry {
    rel: String,
    is_dir: bool,
}
struct Listing {
    entries: Vec<Entry>,
    truncated: bool,
}
fn listing(
    base: &Path,
    kind: &str,
    depth: usize,
    git: Option<&super::process::Output>,
    job: &Job,
) -> anyhow::Result<Listing> {
    ensure!(
        matches!(kind, "file" | "dir" | "all"),
        "invalid type '{kind}'; use file, dir or all"
    );
    ensure!(
        base.is_dir(),
        "search path '{}' is not an accessible directory; choose a readable directory",
        base.display()
    );
    let mut entries = Vec::new();
    let mut used = 0usize;
    let mut truncated = false;
    if let Some(git) = git.filter(|_| kind == "file") {
        truncated = git.truncated;
        for raw in git.stdout.split(|b| *b == 0).filter(|p| !p.is_empty()) {
            job.check()?;
            let rel = String::from_utf8_lossy(raw).into_owned();
            if depth > 0 && Path::new(&rel).components().count() > depth {
                continue;
            }
            if entries.len() >= ENTRY_LIMIT || used.saturating_add(rel.len()) > LIST_BYTES {
                truncated = true;
                break;
            }
            if base.join(&rel).is_file() {
                used += rel.len();
                entries.push(Entry { rel, is_dir: false });
            }
        }
    } else {
        let mut stack = vec![(PathBuf::new(), 0usize)];
        let mut exhausted = false;
        while let Some((relative, level)) = stack.pop() {
            let directory = base.join(relative);
            job.check()?;
            let iter = match fs::read_dir(&directory) {
                Ok(iter) => iter,
                Err(_) => {
                    truncated = true;
                    continue;
                }
            };
            for entry in iter {
                job.check()?;
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(_) => {
                        truncated = true;
                        continue;
                    }
                };
                let path = entry.path();
                let metadata = match fs::metadata(&path) {
                    Ok(m) => m,
                    Err(_) => {
                        truncated = true;
                        continue;
                    }
                };
                let is_dir = metadata.is_dir();
                let rel = path.strip_prefix(base)?.to_string_lossy().into_owned();
                if entries.len().saturating_add(stack.len()) >= ENTRY_LIMIT
                    || used.saturating_add(rel.len()) > LIST_BYTES
                {
                    truncated = true;
                    exhausted = true;
                    break;
                }
                used += rel.len();
                if (kind == "all")
                    || (kind == "dir" && is_dir)
                    || (kind == "file" && metadata.is_file())
                {
                    entries.push(Entry {
                        rel: rel.clone(),
                        is_dir,
                    });
                }
                if is_dir
                    && (depth == 0 || level + 1 < depth)
                    && !entry.file_type()?.is_symlink()
                    && !matches!(
                        entry
                            .file_name()
                            .to_string_lossy()
                            .to_ascii_lowercase()
                            .as_str(),
                        ".git"
                            | ".svn"
                            | ".hg"
                            | "node_modules"
                            | "__pycache__"
                            | ".venv"
                            | "venv"
                            | "dist"
                            | "build"
                            | "target"
                            | ".cache"
                            | ".idea"
                            | ".vscode"
                    )
                {
                    stack.push((PathBuf::from(rel), level + 1));
                }
            }
            if exhausted
                || used > LIST_BYTES
                || entries.len().saturating_add(stack.len()) >= ENTRY_LIMIT
            {
                truncated = true;
                break;
            }
        }
    }
    entries.sort_by(|a, b| a.rel.cmp(&b.rel));
    entries.dedup_by(|a, b| a.rel == b.rel);
    Ok(Listing { entries, truncated })
}
struct Glob {
    pattern: glob::Pattern,
    basename: bool,
}
impl Glob {
    fn new(pattern: &str) -> anyhow::Result<Self> {
        let basename = !pattern.contains('/');
        let pattern = if basename || pattern.starts_with("**/") {
            pattern.to_owned()
        } else if let Some(pattern) = pattern.strip_prefix('/') {
            pattern.to_owned()
        } else {
            format!("**/{pattern}")
        };
        Ok(Self {
            pattern: glob::Pattern::new(&pattern).with_context(|| {
                format!("invalid glob '{pattern}'; check brackets and wildcard syntax")
            })?,
            basename,
        })
    }
    fn matches(&self, path: &str) -> bool {
        let path = if self.basename {
            path.rsplit('/').next().unwrap_or(path)
        } else {
            path
        };
        self.pattern.matches_with(
            path,
            glob::MatchOptions {
                case_sensitive: true,
                require_literal_separator: true,
                require_literal_leading_dot: false,
            },
        )
    }
}
fn search(
    tools: &Tools,
    args: SearchArgs,
    cwd: &Path,
    git: Option<&super::process::Output>,
    job: &Job,
) -> anyhow::Result<Value> {
    ensure!(
        args.limit > 0,
        "invalid limit 0; choose at least one result"
    );
    let include = Glob::new(&args.include)?;
    let exclude = if args.exclude.is_empty() {
        None
    } else {
        Some(Glob::new(&args.exclude)?)
    };
    let base = tools.resolve(cwd, &args.path);
    let listing = listing(&base, &args.kind, args.max_depth, git, job)?;
    let matches = listing
        .entries
        .iter()
        .filter(|e| {
            include.matches(&e.rel)
                && !exclude
                    .as_ref()
                    .is_some_and(|exclude| exclude.matches(&e.rel))
        })
        .collect::<Vec<_>>();
    let shown = matches.len().min(args.limit.min(100));
    let mut text = String::new();
    let mut entries = Vec::new();
    for entry in &matches[..shown] {
        text.push_str(&entry.rel);
        if entry.is_dir {
            text.push('/');
        }
        text.push('\n');
        entries
            .push(serde_json::json!({"path":entry.rel,"type":if entry.is_dir{"dir"}else{"file"}}));
    }
    text.push_str(&format!("\n---\nTotal matches: {}\n", matches.len()));
    if matches.len() > shown {
        text.push_str(&format!("[{shown} results limit reached ({} total matches). Refine the glob pattern to narrow the search.]\n",matches.len()));
    }
    if listing.truncated {
        text.push_str("[results truncated: entry/byte limit or unreadable directory]\n");
    }
    Ok(serde_json::json!({"plain_text_response":text,"entries":entries,"base":base}))
}
enum SearchPattern {
    Linear(regex::Regex),
    Backtracking {
        regex: fancy_regex::Regex,
        work: usize,
    },
}
impl SearchPattern {
    fn new(source: &str, ignore_case: bool) -> anyhow::Result<Self> {
        if let Ok(regex) = regex::RegexBuilder::new(source)
            .case_insensitive(ignore_case)
            .size_limit(2 * 1024 * 1024)
            .dfa_size_limit(2 * 1024 * 1024)
            .build()
        {
            return Ok(Self::Linear(regex));
        }
        let tree = fancy_regex::Expr::parse_tree(source)
            .context("invalid grep regex; check syntax or select literal=true")?;
        ensure!(
            !tree.contains_subroutines,
            "recursive regex subroutines exceed this tool's bounded ECMAScript-style matching contract; simplify the pattern"
        );
        fn complexity(expr: &fancy_regex::Expr) -> anyhow::Result<(usize, usize)> {
            ensure!(
                !matches!(
                    expr,
                    fancy_regex::Expr::Absent(_)
                        | fancy_regex::Expr::BackrefWithRelativeRecursionLevel { .. }
                ),
                "recursive/absent regex extensions are unsupported; use ordinary backreferences or lookarounds"
            );
            let (mut nodes, mut work) = (1usize, 1usize);
            for child in expr.children_iter() {
                let (n, w) = complexity(child)?;
                nodes = nodes.saturating_add(n);
                work = work.saturating_add(w);
            }
            if let fancy_regex::Expr::Repeat { lo, hi, .. } = expr {
                let repeats = if *hi == usize::MAX {
                    (*lo).max(1)
                } else {
                    (*hi).max(1)
                };
                work = work.saturating_mul(repeats);
            }
            ensure!(
                nodes <= 65536 && work <= 65536,
                "grep regex exceeds the expression work budget; reduce repetitions or simplify the pattern"
            );
            Ok((nodes, work))
        }
        let (nodes, work) = complexity(&tree.expr)?;
        // Each AST node can introduce at most one regular delegate in this
        // non-recursive subset. Partition one total NFA/DFA budget across them.
        let delegate_bytes = (2 * 1024 * 1024) / (nodes + 1);
        let regex=fancy_regex::RegexBuilder::new(source).case_insensitive(ignore_case).backtrack_limit(1_000_000).delegate_size_limit(delegate_bytes).delegate_dfa_size_limit(delegate_bytes).build().context("grep regex exceeds its compiled budget or has invalid syntax; simplify the pattern")?;
        Ok(Self::Backtracking { regex, work })
    }
    fn is_match(&self, line: &str) -> anyhow::Result<bool> {
        match self {
            Self::Linear(regex) => Ok(regex.is_match(line)),
            Self::Backtracking { regex, work } => {
                ensure!(
                    work.saturating_mul(line.len().saturating_add(1)) <= 65536,
                    "grep regex/input exceeds its work budget; simplify the pattern or choose literal=true"
                );
                regex.is_match(line).context("grep regex exceeded its backtracking budget; simplify the pattern or choose literal=true")
            }
        }
    }
}

fn grep(
    tools: &Tools,
    args: GrepArgs,
    cwd: &Path,
    git: Option<&super::process::Output>,
    job: &Job,
) -> anyhow::Result<Value> {
    ensure!(
        args.pattern.len() <= 64 * 1024,
        "grep pattern exceeds 65536 bytes; narrow the pattern or search a smaller literal"
    );
    let pattern = if args.literal {
        regex::escape(&args.pattern)
    } else {
        args.pattern
    };
    let pattern = SearchPattern::new(&pattern, args.ignore_case)?;
    let include = Glob::new(&args.include)?;
    let exclude = if args.exclude.is_empty() {
        None
    } else {
        Some(Glob::new(&args.exclude)?)
    };
    let base = tools.resolve(cwd, &args.path);
    let mut listing = if base.is_file() {
        Listing {
            entries: vec![Entry {
                rel: args.path.clone(),
                is_dir: false,
            }],
            truncated: false,
        }
    } else {
        listing(&base, "file", 0, git, job)?
    };
    let single = base.is_file();
    let mut text = String::new();
    let mut total = 0;
    let mut limit = false;
    'files: for entry in &listing.entries {
        job.check()?;
        if !single
            && (!include.matches(&entry.rel)
                || exclude.as_ref().is_some_and(|g| g.matches(&entry.rel)))
        {
            continue;
        }
        let path = if single {
            base.clone()
        } else {
            base.join(&entry.rel)
        };
        let bytes = match read_bounded(&path, FILE_BYTES) {
            Ok(bytes) => bytes,
            Err(_) => {
                listing.truncated = true;
                continue;
            }
        };
        let content = String::from_utf8_lossy(&bytes);
        for (index, line) in content.split_terminator('\n').enumerate() {
            job.check()?;
            if total >= 100 {
                limit = true;
                break 'files;
            }
            if !pattern.is_match(line)? {
                continue;
            }
            let start = index.saturating_sub(args.context_lines);
            let end = index.saturating_add(args.context_lines);
            for (j, line) in content
                .split_terminator('\n')
                .enumerate()
                .skip(start)
                .take(end.saturating_sub(start).saturating_add(1))
            {
                job.check()?;
                let separator = if j == index { ':' } else { '-' };
                let number = if args.return_line_numbers || args.context_lines > 0 {
                    format!("{}{}", j + 1, separator)
                } else {
                    String::new()
                };
                let n = entry
                    .rel
                    .len()
                    .saturating_add(number.len())
                    .saturating_add(line.len())
                    .saturating_add(2);
                if text.len().saturating_add(n) > LIST_BYTES {
                    limit = true;
                    break 'files;
                }
                text.push_str(&entry.rel);
                text.push(separator);
                text.push_str(&number);
                text.push_str(line);
                text.push('\n');
            }
            if args.context_lines > 0 {
                text.push_str("--\n");
            }
            total += 1;
        }
    }
    text.push_str(&format!("\n---\nTotal matches: {total}\n"));
    if limit {
        text.push_str(
            "[matches/output limit reached. Narrow the path/pattern/include to see more.]\n",
        );
    }
    if listing.truncated {
        text.push_str("[results truncated: unreadable or oversized files]\n");
    }
    Ok(serde_json::json!({"plain_text_response":text}))
}

fn normalize(content: &str) -> anyhow::Result<String> {
    let mut result = String::new();
    result
        .try_reserve_exact(content.len())
        .context("reserve normalized edit text")?;
    for (index, line) in content.split('\n').enumerate() {
        if index != 0 {
            result.push('\n');
        }
        for c in line.trim_end_matches([' ', '\t', '\r']).chars() {
            result.push(match c {
                '\u{2018}'..='\u{201b}' => '\'',
                '\u{201c}'..='\u{201f}' => '"',
                '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
                '\u{00a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
                _ => c,
            });
        }
    }
    Ok(result)
}
struct Replacement {
    start: usize,
    end: usize,
    new: String,
}
fn apply(content: &str, replacements: &[Replacement], offset: usize) -> String {
    let mut result = content.to_owned();
    for r in replacements.iter().rev() {
        result.replace_range(r.start - offset..r.end - offset, &r.new);
    }
    result
}
fn edited(content: &str, edits: Vec<Edit>) -> anyhow::Result<String> {
    ensure!(!edits.is_empty(), "edits must be a non-empty array");
    let fuzzy = edits.iter().any(|edit| !content.contains(&edit.old_text));
    let normalized = normalize(content)?;
    let base = if fuzzy { normalized.as_str() } else { content };
    let mut replacements = Vec::new();
    for (index, edit) in edits.into_iter().enumerate() {
        ensure!(
            !edit.old_text.is_empty(),
            "edits[{index}].old_text must not be empty"
        );
        let norm = normalize(&edit.old_text)?;
        let occurrences = if norm.is_empty() {
            content.matches(&edit.old_text).count()
        } else {
            normalized.matches(&norm).count()
        };
        ensure!(
            occurrences == 1,
            "edits[{index}].old_text has {occurrences} matches; supply text that uniquely matches the current file"
        );
        let old = if fuzzy {
            norm.as_str()
        } else {
            edit.old_text.as_str()
        };
        ensure!(
            !old.is_empty(),
            "edits[{index}] disappears after whitespace normalization; replace the complete line instead"
        );
        let start = base.find(old).with_context(|| {
            format!(
                "could not find edits[{index}].old_text; match the current file content exactly"
            )
        })?;
        replacements.push(Replacement {
            start,
            end: start + old.len(),
            new: edit.new_text,
        });
    }
    replacements.sort_by_key(|r| r.start);
    ensure!(
        replacements.windows(2).all(|r| r[0].end <= r[1].start),
        "edits overlap; merge nearby changes or target disjoint original regions"
    );
    let result = if !fuzzy {
        apply(base, &replacements, 0)
    } else {
        let line_count = content.bytes().filter(|b| *b == b'\n').count()
            + usize::from(!content.is_empty() && !content.ends_with('\n'));
        let mut originals = Vec::new();
        originals
            .try_reserve_exact(line_count)
            .context("reserve original edit line descriptors")?;
        originals.extend(content.split_inclusive('\n'));
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(line_count)
            .context("reserve normalized edit line descriptors")?;
        let mut offset = 0;
        for line in base.split_inclusive('\n') {
            spans.push((offset, offset + line.len()));
            offset += line.len();
        }
        let mut groups: Vec<(usize, usize, Vec<Replacement>)> = Vec::new();
        for r in replacements {
            let start = spans
                .iter()
                .position(|(a, b)| r.start >= *a && r.start < *b)
                .context(
                    "normalized edit starts outside line boundaries; replace the full file instead",
                )?;
            let end = spans
                .iter()
                .position(|(_, b)| *b >= r.end)
                .context("normalized edit ends outside line boundaries")?
                + 1;
            if let Some(group) = groups.last_mut().filter(|g| start < g.1) {
                group.1 = group.1.max(end);
                group.2.push(r);
            } else {
                groups.push((start, end, vec![r]));
            }
        }
        let mut result = String::new();
        let mut cursor = 0;
        for (start, end, replacements) in groups {
            for line in &originals[cursor..start] {
                result.push_str(line);
            }
            result.push_str(&apply(
                &base[spans[start].0..spans[end - 1].1],
                &replacements,
                spans[start].0,
            ));
            cursor = end;
        }
        for line in &originals[cursor..] {
            result.push_str(line);
        }
        result
    };
    ensure!(
        result != content,
        "no changes made: replacements produced identical content"
    );
    ensure!(
        result.len() <= FILE_BYTES,
        "edited file exceeds the {FILE_BYTES} byte limit; split the edit or choose a smaller file"
    );
    Ok(result)
}

pub(super) fn invoke(
    tools: &Tools,
    tool: BuiltinTool,
    params: &str,
    cwd: &Path,
    base64: bool,
    git: Option<(Option<super::process::Output>, MemoryLease)>,
    job: Arc<Job>,
) -> anyhow::Result<ToolValue> {
    job.check()?;
    let mut leases = Vec::new();
    let git_output = git.as_ref().and_then(|g| g.0.as_ref());
    let value = match tool {
        BuiltinTool::ReadFile => {
            let (value, credit) = text_read(
                tools,
                serde_json::from_str(params).context("invalid read_file parameters")?,
                cwd,
                base64,
                &job,
            )?;
            leases.extend(credit);
            value
        }
        BuiltinTool::WriteFile => {
            let args: WriteArgs =
                serde_json::from_str(params).context("invalid write_file parameters")?;
            ensure!(
                args.content.len() <= FILE_BYTES,
                "write content exceeds the {FILE_BYTES} byte limit"
            );
            let _lock = mutation_lock(tools, &job)?;
            atomic_write(
                &tools.resolve(cwd, &args.path),
                args.content.as_bytes(),
                &job,
            )?;
            serde_json::json!({"result":"file written successfully","path":args.path,"bytes":args.content.len()})
        }
        BuiltinTool::EditFile => {
            let args: EditArgs =
                serde_json::from_str(params).context("invalid edit_file parameters")?;
            let _lock = mutation_lock(tools, &job)?;
            let path = tools.resolve(cwd, &args.path);
            let length = usize::try_from(
                fs::metadata(&path)
                    .with_context(|| {
                        format!(
                            "inspect edit_file target '{}'; verify the path exists and is readable",
                            path.display()
                        )
                    })?
                    .len(),
            )?;
            leases.push(
                tools.reserve(
                    length
                        .checked_mul(40)
                        .context("edit staging size overflow")?,
                )?,
            );
            let bytes = read_bounded(&path, FILE_BYTES)?;
            let original = std::str::from_utf8(&bytes).context(
                "edit_file requires UTF-8 text; use write_file to replace binary content",
            )?;
            let count = args.edits.len();
            let result = edited(original, args.edits)?;
            atomic_write(&path, result.as_bytes(), &job)?;
            serde_json::json!({"result":"file edited successfully","path":args.path,"edits_applied":count})
        }
        BuiltinTool::FileGlobSearch => {
            if git.is_none() {
                leases.push(tools.reserve(LIST_BYTES * 8)?);
            }
            search(
                tools,
                serde_json::from_str(params).context("invalid file_glob_search parameters")?,
                cwd,
                git_output,
                &job,
            )?
        }
        BuiltinTool::GrepSearch => {
            leases.push(tools.reserve(LIST_BYTES * 16)?);
            grep(
                tools,
                serde_json::from_str(params).context("invalid grep_search parameters")?,
                cwd,
                git_output,
                &job,
            )?
        }
        _ => anyhow::bail!("tool '{}' is not a filesystem operation", tool.name()),
    };
    if let Some((_, lease)) = git {
        leases.push(lease);
    }
    Ok(ToolValue { value, leases, job })
}
